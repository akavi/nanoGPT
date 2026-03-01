from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from src.models.nn import Linear, Normalization

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x


def ste_func(x):
    return STE.apply(x)


class SmoothingModuleState(TypedDict):
    previous_value: torch.Tensor


class DeTokenizer(nn.Module):
    def __init__(
        self,
        d_model=1024,
        residual_in_fp32: bool = True,
        ema_in_fp32: bool = False,
        d_head=32,
        mode: str = "v1",
        attn_norm: str = "rms",
        out_norm: str = "",
        chunk_size=256,
        epsilon=1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        assert d_model % self.d_head == 0
        self.n_heads = d_model // self.d_head
        self.chunk_size = chunk_size
        self.epsilon = epsilon

        self.residual_in_fp32 = residual_in_fp32
        self.ema_in_fp32 = ema_in_fp32

        self.mode = mode
        assert self.mode in ("v1", "v2")
        if self.mode == "v2":
            self.attn_norm = (
                Normalization(self.d_model, _name_=out_norm) if attn_norm else nn.Identity()
            )
            self.q_proj = Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = Linear(self.d_model, self.d_model, bias=False)

        self.out_norm = Normalization(self.d_model, _name_=out_norm) if out_norm else None

    def forward(
        self, hidden_states, residual, token_mask, router_probs,
        x_pack_kwargs: torch.Tensor = None, state=None,
    ):
        assert hidden_states.shape[0] == 1, "Hidden states must have batch dimension 1"
        assert token_mask.shape[0] == 1, "Token mask must have batch dimension 1"
        assert router_probs.shape[0] == 1, "Router probs must have batch dimension 1"

        upsampler_out, state = self.forward_packed(
            hidden_states, token_mask, router_probs, x_pack_kwargs, state
        )

        # x: long_state (B, L, D)
        if self.mode == "v1":  # STE
            coef = torch.max(router_probs, dim=-1).values  # (B, L)
            coef = ste_func(coef).to(upsampler_out.dtype)
            out = upsampler_out.mul_(coef.unsqueeze(-1))
        elif self.mode == "v2":
            # in-place changes to upsampler_out
            out = upsampler_out
            out[token_mask] *= router_probs[token_mask][:, [1]].to(out.dtype)

            q_input = residual[~token_mask]
            k_input = upsampler_out[~token_mask]
            qk = torch.einsum(
                "l d, l d -> l",
                self.q_proj(self.attn_norm(q_input)),
                self.k_proj(self.attn_norm(k_input)),
            ) / (self.d_model**0.5)
            coef = torch.sigmoid(F.silu(qk).to(torch.float32)).to(qk.dtype)
            out[~token_mask] *= coef.unsqueeze(-1)
        else:
            raise NotImplementedError

        out_dtype = out.dtype
        if self.residual_in_fp32:
            out = out.to(torch.float32)
        out = (residual + out).to(out_dtype)

        if self.out_norm:
            out = self.out_norm(out)

        return out, state

    def forward_packed(self, hidden_states, token_mask, router_probs, x_pack_kwargs, state):
        # Raw boundary probability of taking the main-stage path (kept unclamped so gradients flow).
        # boundary_prob: (1, L, [no-boundary, boundary]) -> raw_prob: (1, M) (prob of boundary)
        p = router_probs[..., -1].float()  # (1, L)
        chunk_probs = p[token_mask].unsqueeze(0)  # (1, M)

        if state is not None:
            N = x_pack_kwargs.shape[1]
            chunk_counts = x_pack_kwargs[0]  # (N,)
            max_per_seq = int(chunk_counts.max().item())
            device = hidden_states.device

            # Unpack chunks from (1, M, D) to (N, max_per_seq, D) for per-sequence EMA
            unpacked_chunks = hidden_states.new_zeros(N, max_per_seq, self.d_model)
            unpacked_probs = chunk_probs.new_zeros(N, max_per_seq)
            valid_mask = torch.arange(max_per_seq, device=device)[None, :] < chunk_counts[:, None]
            unpacked_chunks[valid_mask] = hidden_states[0]
            unpacked_probs[valid_mask] = chunk_probs[0]

            # Per-sequence initial states: (1, N, D) -> (N, D)
            per_seq_state = SmoothingModuleState(
                previous_value=state["previous_value"].squeeze(0)
            )

            # Run EMA with N as batch dim
            out_unpacked = self.ema(unpacked_chunks, unpacked_probs, state=per_seq_state)

            # Repack to (1, M, D)
            out = out_unpacked[valid_mask].unsqueeze(0)

            # State update: last EMA output per sequence
            last_positions = (chunk_counts - 1).clamp(min=0)  # (N,)
            last_ema_per_seq = out_unpacked[torch.arange(N, device=device), last_positions]  # (N, D)
            # Preserve state for sequences with no chunks
            no_chunks = chunk_counts == 0
            if no_chunks.any():
                last_ema_per_seq[no_chunks] = state["previous_value"].squeeze(0)[no_chunks]
            new_state = SmoothingModuleState(previous_value=last_ema_per_seq.unsqueeze(0))  # (1, N, D)
        else:
            out = self.ema(hidden_states, chunk_probs, state=None)
            new_state = None

        plug_back_idx = torch.cumsum(token_mask, dim=-1) - 1
        long_states = torch.gather(
            out, dim=1, index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        return long_states, new_state

    def forward_unpacked(self, hidden_states, token_mask, router_probs, state):
        B, M, D = hidden_states.shape
        B, L = token_mask.shape

        # Handle empty sequence case
        if L == 0:
            out = torch.empty(B, 0, D, device=hidden_states.device, dtype=hidden_states.dtype)
            return out

        assert router_probs is not None, "router_probs must be provided for ema mode"

        # Raw boundary probability of taking the main-stage path (kept unclamped so gradients flow).
        # boundary_prob: (B, L, [no-boundary, boundary]) -> raw_prob: (B, L) (prob of boundary)
        boundary_prob = router_probs[..., -1].float()  # (B, L)

        # Compute chunk indices via cumsum (reuse for multiple operations)
        # chunk_idx[i] = (index of chunk that position i belongs to), or -1 if no chunk yet
        chunk_idx = torch.cumsum(token_mask.to(torch.long), dim=1) - 1  # (B, L)
        valid_positions = chunk_idx >= 0  # (B, L)

        # Count tokens per sequence - derive from cumsum (last valid value + 1)
        # This avoids a separate sum() call
        num_tokens = chunk_idx[:, -1] + 1  # (B,) - cumsum at end equals total count

        # Clamp chunk_idx for safe indexing (invalid positions will be zeroed out later)
        chunk_idx_clamped = chunk_idx.clamp(min=0, max=M - 1)

        # Create valid_chunk_mask for hidden_states (B, M)
        valid_chunk_mask = (
            torch.arange(M, device=hidden_states.device)[None, :] < num_tokens[:, None]
        )  # (B, M)

        # For chunk_probs, we need to gather boundary_prob at token positions.
        # Instead of argsort, use scatter_add to accumulate probs at their chunk indices.
        # Since each chunk has exactly one token, and non-token positions have zero prob,
        # scatter_add gives the correct result (adding zeros doesn't change the sum).
        # chunk_probs[b, chunk_idx[b, i]] = boundary_prob[b, i] where token_mask[b, i] = True

        # Create chunk_probs via scatter_add (avoids argsort, handles overwrites correctly)
        chunk_probs = torch.zeros(B, M, device=hidden_states.device, dtype=boundary_prob.dtype)
        # Mask boundary_prob to only selected positions
        token_mask_float = token_mask.to(boundary_prob.dtype)
        selected_prob = boundary_prob * token_mask_float  # (B, L)
        # Use scatter_add_ so that zeros don't overwrite the actual probabilities
        chunk_probs.scatter_add_(1, chunk_idx_clamped, selected_prob)

        # Zero out invalid chunks (positions beyond num_tokens for each batch)
        # Use in-place multiplication (chunk_probs is freshly created, safe for in-place)
        chunk_probs.mul_(valid_chunk_mask.to(chunk_probs.dtype))

        # Apply valid_chunk_mask to hidden_states
        chunk_states = hidden_states * valid_chunk_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )  # (B, M, D)

        out = self.ema(chunk_states, chunk_probs, state=state)

        # Broadcast chunk states and probs back to original sequence length
        # Note: advanced indexing creates copies, so long_states is fresh tensors
        batch_idx = torch.arange(B, device=hidden_states.device)[:, None]
        long_states = out[batch_idx, chunk_idx_clamped]  # (B, L, D)

        # Zero out invalid positions using in-place multiplication
        # (long_states is fresh from advanced indexing, safe for in-place)
        valid_mask_float = valid_positions.to(long_states.dtype)
        long_states.mul_(valid_mask_float.unsqueeze(-1))  # (B, L, D) in-place

        if state is not None:
            state_add_mask = token_mask.cumsum(dim=1) == 0
            batch_idx = torch.arange(B, device=hidden_states.device)[:, None].expand(-1, L)[
                state_add_mask
            ]
            long_states[state_add_mask] = (
                long_states[state_add_mask] + state["previous_value"][batch_idx]
            )

        return long_states

    def ema(
        self,
        hidden_states: torch.Tensor,
        p: torch.Tensor,
        state: torch.Tensor | None,
    ):
        # Clamp probabilities to avoid log underflow in a differentiable way.
        clipped_p = p.clamp(min=self.epsilon, max=1 - self.epsilon)

        in_dtype = hidden_states.dtype
        ema_dtype = torch.float32 if self.ema_in_fp32 else in_dtype

        dt = torch.log(1 / (1 - clipped_p)).to(ema_dtype)
        x = (hidden_states / dt[..., None]).to(ema_dtype)  # (B, L, D)
        A = -torch.ones((self.n_heads,), device=hidden_states.device, dtype=torch.float32)
        b = clipped_p.to(ema_dtype)
        c = torch.ones_like(b)
        initial_states = (
            rearrange(
                state["previous_value"],
                "b (h p) -> b h p 1",
                p=self.d_head,
            )
            if state is not None
            else None
        )

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.d_head),
            repeat(dt, "b l -> b l h", h=self.n_heads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.chunk_size,
            initial_states=initial_states,
            # seq_idx=seq_idx, # NOTE seq_idx is unnecessary as every first token p == 1.0
        ).to(in_dtype)
        out = rearrange(out, "b l h p -> b l (h p)")

        return out

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        is_token: torch.Tensor,
        p_tokenize: torch.Tensor,
        state: SmoothingModuleState | None,
    ):
        """
        Steps the detokenizer.
        x: (b, D) (b = sum(is_token))
        residual: (B, D)
        is_token: boolean (B,)
        p_tokenize: float (B,)
        """
        assert state is not None, "State is required for detokenizer step"

        p_tokenize = p_tokenize[is_token][:, None]  # (b, 1)

        # Recurrence: h[t] = (1-p) * h[t-1] + p * (2p-1) * x
        new_ema_state = (1 - p_tokenize) * state["previous_value"][is_token] + p_tokenize * x

        ema_state = state["previous_value"].clone()
        ema_state[is_token] = new_ema_state.to(ema_state.dtype)

        if self.mode == "v1":
            # at train time, we do residual + ste(p) * x
            # since ste(p) is 1 in the forward and p in the backward, at inference time, we just do x+residual
            out = ema_state + residual
        elif self.mode == "v2":
            out = residual.clone()  # (B, D)
            # multiply p to tokenized positions
            out[is_token] += p_tokenize * ema_state[is_token]  # (b, D)

            qk = torch.einsum(
                "b d, b d -> b",
                self.q_proj(residual[~is_token]),
                self.k_proj(ema_state[~is_token]),
            ) / (self.d_model**0.5)
            coef = torch.sigmoid(F.silu(qk))  # (B,)

            out[~is_token] += coef.unsqueeze(-1) * ema_state[~is_token]
        else:
            raise NotImplementedError

        if self.out_norm:
            out = self.out_norm(out)

        return out, SmoothingModuleState(previous_value=ema_state)

    def default_state(
        self,
        *batch_shape,
        previous_value=None,
        device=None,
        dtype=None,
        inplace_state=None,
        **kwargs,
    ):
        if inplace_state is not None:
            raise NotImplementedError("Inplace state not yet supported for DeTokenizer")
        if previous_value is None:
            previous_value = torch.zeros(*batch_shape, self.d_model, device=device, dtype=dtype)
        else:
            assert previous_value.shape == (
                batch_shape + (self.d_model,)
            ), "Expected previous_value to have shape {} but got {}".format(
                batch_shape + (self.d_model,), previous_value.shape
            )
        return SmoothingModuleState(previous_value=previous_value)


class Tokenizer(nn.Module):
    def __init__(
        self,
        selector=None,
    ):
        super().__init__()
        self.selector = selector

    def forward(self, hidden_states, token_mask, x_pack_kwargs: torch.Tensor = None, router_probs=None):
        """Given a token mask, selects and compacts hidden states corresponding to True positions.

        Args:
            hidden_states: (1, L, D)
            token_mask: (1, L) boolean mask indicating which positions to select
            x_pack_kwargs: Tensor of per-sequence lengths, shape (1, N)
            router_probs: (1, L, 2) unused in forward, kept for interface consistency

        Returns:
            Tuple of (chunked_hidden_states, chunked_token_mask, chunked_pack_kwargs)
            - chunked_hidden_states: (1, M, D) where M is total selected tokens
            - chunked_token_mask: None (always packed)
            - chunked_pack_kwargs: Tensor of chunked per-sequence lengths, shape (1, N)
        """
        assert hidden_states.shape[0] == 1, "Packed input must have batch dimension 1"
        chunked_hidden_states = hidden_states[token_mask].unsqueeze(0)

        lens_cs = F.pad(x_pack_kwargs[0].cumsum(0), (1, 0)).long()
        chunked_lens_cs = F.pad(
            token_mask.cumsum(dim=1)[0, lens_cs[1:] - 1], (1, 0)
        )
        chunked_lens = chunked_lens_cs[1:] - chunked_lens_cs[:-1]
        chunked_pack_kwargs = chunked_lens.unsqueeze(0)  # (1, N)

        return chunked_hidden_states, None, chunked_pack_kwargs

    def step(self, x, token_mask: torch.Tensor, router_probs: torch.Tensor, **kwargs):
        return x  # We'll let downstream handle whether to run the main stage or not
