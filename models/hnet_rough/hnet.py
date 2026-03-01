"""DynamicTokenizerModel and CausalLM wrapper.

All sub-modules are accepted as pre-built constructor arguments rather than
being constructed internally from config dicts.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import torch
import torch.nn as nn
from optree import tree_map
from torch import Tensor

from src.models.functional.pack import pack, unpack
from src.models.sequence.hnet.router import Router, RouterState
from src.models.sequence.hnet.stage import Stage
from src.models.sequence.hnet.tokenizer import DeTokenizer, SmoothingModuleState, Tokenizer
from src.utils.data import DATA_PREFIX

ROUTER_OUTPUT_KEY = f"{DATA_PREFIX}router_output"
ROUTER_OUTPUT_PACKED_KEY = f"{DATA_PREFIX}router_output_packed"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

class DynamicTokenizerState(TypedDict):
    pre_stage: Any
    main_stage: Any
    post_stage: Any
    router: RouterState | None
    detokenizer: SmoothingModuleState | None


def collect_state(state: Any, mask: torch.Tensor) -> Any:
    """Return a new state keeping only elements where *mask* is True."""
    return tree_map(lambda x: x[mask], state)


def return_state(state: Any, mask: torch.Tensor, new_state: Any) -> Any:
    """Scatter *new_state* back into *state* at positions given by *mask*."""
    return tree_map(lambda x, y: x.index_copy(0, mask, y.to(x.dtype)), state, new_state)


# ---------------------------------------------------------------------------
# Layout DSL
# ---------------------------------------------------------------------------

def split_not_in_parens(to_split: str, split_char: str) -> list[str]:
    """Split *to_split* by *split_char*, respecting parenthesised groups."""
    parts: list[str] = []
    paren_depth = 0
    current = ""
    for c in to_split:
        if c == "(":
            paren_depth += 1
            current += c
        elif c == ")":
            paren_depth -= 1
            current += c
        elif c == split_char and paren_depth == 0:
            parts.append(current)
            current = ""
        else:
            current += c
    return parts + [current]


def parse_layer_layout(layer_layout: str) -> list[str]:
    """Parse a layout DSL string into a flat list of layer names.

    Examples::

        "ssd-3-sa-1"          -> ["ssd", "ssd", "ssd", "sa"]
        "(ssd-1-mqa-1)-2"     -> ["ssd", "mqa", "ssd", "mqa"]
    """
    split_parts = split_not_in_parens(layer_layout, "-")
    assert len(split_parts) % 2 == 0, "Split parts must be even length"

    result: list[str] = []
    for i in range(0, len(split_parts), 2):
        sublayout = split_parts[i]
        reps = int(split_parts[i + 1])
        if sublayout.startswith("(") and sublayout.endswith(")"):
            sublayout_parsed = parse_layer_layout(sublayout[1:-1])
        else:
            sublayout_parsed = [sublayout]
        result.extend(sublayout_parsed * reps)
    return result


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module, n_layer, initializer_range=0.02):
    """Initialise weights of *module* (called recursively via ``nn.apply``)."""
    if isinstance(module, nn.Linear) and not getattr(module.weight, "_no_reinit", False):
        nn.init.normal_(module.weight, std=initializer_range)
        optim_cfg = getattr(module.weight, "_optim", {})
        setattr(module.weight, "_optim", optim_cfg)

        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)

    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
            if "out_proj" in name or "out_linear" in name:
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range / math.sqrt(n_layer))


# ---------------------------------------------------------------------------
# DynamicTokenizerModel
# ---------------------------------------------------------------------------

class DynamicTokenizerModel(nn.Module):
    """Dynamic Tokenizer Model.

    All children are pre-built and passed in via the constructor (DI-style).
    """

    def __init__(
        self,
        d_model: int,
        pre_stage: Stage,
        main_stage: Stage | DynamicTokenizerModel,
        post_stage: Stage,
        router: Router,
        tokenizer: Tokenizer,
        detokenizer: DeTokenizer,
        residual_proj: nn.Module,
        pad_parameter: nn.Parameter | None = None,
        residual_in_fp32: bool = True,
        track_flops: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pre_stage = pre_stage
        self.main_stage = main_stage
        self.post_stage = post_stage
        self.router = router
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.residual_proj = residual_proj
        self.residual_in_fp32 = residual_in_fp32
        self.track_flops = track_flops

        if pad_parameter is not None:
            self.pad_parameter = pad_parameter
        else:
            self.pad_parameter = None

        self.metrics: dict[str, Any] = {}

    def n_residual(self) -> list[int | list]:
        return [
            self.pre_stage.n_residual(),
            self.main_stage.n_residual(),
            self.post_stage.n_residual(),
        ]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: Tensor,
        x_pack_kwargs: Tensor,
        state: DynamicTokenizerState | None = None,
        return_after_pre_stage: bool = False,
        **mixer_kwargs: Any,
    ) -> tuple[Tensor, DynamicTokenizerState | None, dict[str, Any]]:
        self.metrics = {"flops": torch.tensor(0, device=hidden_states.device, dtype=torch.long)}

        assert len(hidden_states.shape) == 3, "Hidden states must be 3D"

        B, L, D = hidden_states.shape
        assert B == 1, "Packed input must have batch dimension 1"

        if state is not None:
            prev_pre_stage_state = state["pre_stage"]
            prev_main_stage_state = state["main_stage"]
            prev_post_stage_state = state["post_stage"]
            prev_router_state = state["router"]
            prev_detokenizer_state = state["detokenizer"]
        else:
            prev_pre_stage_state = None
            prev_main_stage_state = None
            prev_post_stage_state = None
            prev_router_state = None
            prev_detokenizer_state = None

        # Run the pre-stage
        hidden_states, pre_stage_state, _ = self.pre_stage(
            hidden_states, x_pack_kwargs=x_pack_kwargs,
            state=prev_pre_stage_state, **mixer_kwargs,
        )

        # Apply the residual projection (in float32 for numerical stability)
        if self.residual_in_fp32:
            with torch.autocast(device_type="cuda", enabled=False):
                residual = self.residual_proj(hidden_states.float())
        else:
            residual = self.residual_proj(hidden_states)

        self.metrics.update({"norm/outer_res": torch.mean(residual.detach() ** 2)})

        # Run the router to compute chunking probabilities
        router_output, router_state = self.router.forward(
            hidden_states,
            x_pack_kwargs=x_pack_kwargs,
            state=prev_router_state,
        )

        # return router outputs without running main stage
        if return_after_pre_stage:
            return (
                hidden_states,
                None,
                {f"{DATA_PREFIX}main_stage_router_output": [router_output]},
            )

        # Tokenize
        hidden_states, _, chunked_pack_kwargs = self.tokenizer.forward(
            hidden_states,
            router_output.token_mask,
            router_probs=router_output.router_probs,
            x_pack_kwargs=x_pack_kwargs,
        )

        B, M, D = hidden_states.shape
        if self.pad_parameter is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_parameter[None, None, :].expand(B, M, -1)),
                dim=-1,
            )

        # Run the main stage
        hidden_states, main_stage_state, main_kwargs = self.main_stage(
            hidden_states,
            x_pack_kwargs=chunked_pack_kwargs,
            state=prev_main_stage_state,
            **mixer_kwargs,
        )
        inner_router_output = main_kwargs.get(ROUTER_OUTPUT_KEY, [])
        inner_router_output_packed = main_kwargs.get(ROUTER_OUTPUT_PACKED_KEY, [])

        # Truncate the hidden states to the original dimension
        hidden_states = hidden_states[..., :D]

        # Run the detokenizer
        hidden_states, detokenizer_state = self.detokenizer.forward(
            hidden_states,
            residual,
            router_output.token_mask,
            router_output.router_probs,
            state=prev_detokenizer_state,
            x_pack_kwargs=chunked_pack_kwargs,
        )

        self.metrics.update({"norm/outer_res_fuse": torch.mean(hidden_states.detach() ** 2)})

        # Run the post-stage
        hidden_states, post_stage_state, _ = self.post_stage(
            hidden_states, x_pack_kwargs=x_pack_kwargs,
            state=prev_post_stage_state, **mixer_kwargs,
        )

        self._consolidate_metrics(router_output)
        self._add_flops(B, L)

        state = DynamicTokenizerState(
            pre_stage=pre_stage_state,
            main_stage=main_stage_state,
            post_stage=post_stage_state,
            router=router_state,
            detokenizer=detokenizer_state,
        )
        return (
            hidden_states,
            state,
            {
                ROUTER_OUTPUT_KEY: [router_output] + inner_router_output,
                ROUTER_OUTPUT_PACKED_KEY: [router_output] + inner_router_output_packed,
            },
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self, x: Tensor, state: DynamicTokenizerState, period_idx: int | None = None, **kwargs
    ):
        # step the pre-stage
        x, pre_stage_state = self.pre_stage.step(x, state["pre_stage"], **kwargs)

        # step the router
        router_probs, is_token, selected_probs, router_state = self.router.step(
            x, state["router"], **kwargs
        )

        # take residual (in float32 for numerical stability), then step the tokenizer
        input_dtype = x.dtype
        residual = self.residual_proj(x.to(torch.float32)).to(input_dtype)
        x = self.tokenizer.step(x, is_token, router_probs, **kwargs)

        B, D = x.shape

        # pad the hidden states if needed to the deeper dimension
        if self.pad_parameter is not None:
            x = torch.cat((x, self.pad_parameter[None, :].expand(B, -1)), dim=-1)

        if is_token.any():
            main_stage_state = collect_state(state["main_stage"], is_token)
            x, new_main_stage_state = self.main_stage.step(x[is_token], main_stage_state, **kwargs)

            main_stage_state = return_state(
                state["main_stage"], torch.where(is_token)[0], new_main_stage_state
            )
        else:
            main_stage_state = state["main_stage"]
            x = x[is_token]  # (b, D) but b = 0

        # truncate the hidden states to the original dimension
        x = x[..., :D]

        # step the detokenizer
        x, detokenizer_state = self.detokenizer.step(
            x, residual, is_token, router_probs, state["detokenizer"], **kwargs
        )

        x, post_stage_state = self.post_stage.step(x, state["post_stage"], **kwargs)

        return x, DynamicTokenizerState(
            pre_stage=pre_stage_state,
            main_stage=main_stage_state,
            post_stage=post_stage_state,
            router=router_state,
            detokenizer=detokenizer_state,
        )

    # ------------------------------------------------------------------
    # default_state
    # ------------------------------------------------------------------
    def default_state(
        self, *batch_shape, device=None, dtype=None, **kwargs
    ) -> DynamicTokenizerState:
        assert len(batch_shape) == 1, "Default state not yet supported for multidimensional batch shapes"

        return DynamicTokenizerState(
            pre_stage=self.pre_stage.default_state(
                *batch_shape, device=device, dtype=dtype, **kwargs
            ),
            main_stage=self.main_stage.default_state(
                *batch_shape, device=device, dtype=dtype, **kwargs
            ),
            post_stage=self.post_stage.default_state(
                *batch_shape, device=device, dtype=dtype, **kwargs
            ),
            router=self.router.default_state(*batch_shape, device=device, dtype=dtype, **kwargs),
            detokenizer=self.detokenizer.default_state(
                *batch_shape, device=device, dtype=dtype, **kwargs
            ),
        )

    # ------------------------------------------------------------------
    # Stubs for compatibility
    # ------------------------------------------------------------------
    def set_lora_and_forward(self, *args, lora=None, **kwargs):
        return self.forward(*args, **kwargs)

    def cg_built(self, method_name: str) -> bool:
        return False

    def set_peft_and_forward(self, inputs, *args, state=None, peft_state=None, peft_type=None, **kwargs):
        return self.forward(inputs, *args, state=state, **kwargs)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _add_metrics(self, prefix, metrics):
        for k, v in metrics.items():
            if k == "flops":
                self.metrics["flops"] += v
            else:
                if k.startswith("norm/"):
                    self.metrics[f"norm/{prefix}/{k[5:]}"] = v
                else:
                    self.metrics[f"{prefix}/{k}"] = v
        metrics.clear()

    def _consolidate_metrics(self, router_output):
        self._add_metrics("pre_stage", self.pre_stage.metrics)
        self._add_metrics("main_stage", self.main_stage.metrics)
        self._add_metrics("post_stage", self.post_stage.metrics)

        self.metrics["p_max"] = router_output.selected_probs.detach().mean()
        self.metrics["token_ratio"] = router_output.token_mask.detach().float().mean()

    def _add_flops(self, B, L):
        residual_proj_param_count = sum(p.numel() for p in self.residual_proj.parameters())
        router_param_count = sum(p.numel() for p in self.router.parameters())
        self.metrics["flops"] += 2 * B * L * (residual_proj_param_count + router_param_count)


class DynamicTokenizerModelWrapper(nn.Module):
    """CausalLM wrapper around DynamicTokenizerModel."""

    def __init__(
        self,
        backbone: DynamicTokenizerModel,
        vocab_size: int,
        d_embed: int,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_embed)
        self.backbone = backbone
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        mask: Tensor | None = None,
        num_tokens: Tensor | None = None,
        state: DynamicTokenizerState | None = None,
        num_last_tokens: int = 0,
        **kwargs: Any,
    ) -> tuple[Tensor, DynamicTokenizerState | None, dict[str, Any]]:
        hidden_states = self.embeddings(input_ids)
        B, L, _ = hidden_states.shape

        # Build mask
        if mask is None:
            if num_tokens is not None:
                mask = torch.arange(L, device=hidden_states.device)[None, :] < num_tokens[:, None]
            else:
                mask = torch.ones((B, L), device=hidden_states.device, dtype=torch.bool)

        # Pack
        packed_x, idx, lens_cs, _ = pack(hidden_states, mask)
        seq_lens = lens_cs[1:] - lens_cs[:-1]  # (N,)
        x_pack_kwargs = seq_lens.unsqueeze(0)  # (1, N)

        # Backbone always receives packed input
        hidden_states, state, metrics = self.backbone(
            packed_x.unsqueeze(0),
            x_pack_kwargs=x_pack_kwargs,
            state=state,
            **kwargs,
        )

        # Unpack
        hidden_states = unpack(hidden_states.squeeze(0), idx, B, L)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        logits = self.lm_head(hidden_states)

        # Add FLOPs for lm_head
        lm_head_seq_len = hidden_states.shape[1]
        d_embed = self.embeddings.embedding_dim
        lm_head_flops = 2 * B * lm_head_seq_len * d_embed * self.vocab_size
        self.backbone.metrics["flops"] += lm_head_flops

        return logits, state, metrics

    def step(self, x: Tensor, state: DynamicTokenizerState, **kwargs):
        hidden_states = self.embeddings(x).squeeze(1)
        hidden_states, state = self.backbone.step(hidden_states, state, **kwargs)
        logits = self.lm_head(hidden_states)
        return logits, state

    def default_state(self, *batch_shape, device=None, dtype=None, **kwargs) -> DynamicTokenizerState:
        return self.backbone.default_state(*batch_shape, device=device, dtype=dtype, **kwargs)
