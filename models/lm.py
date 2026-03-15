"""Generic LM wrappers: PatchLM (linear embedding) and CategoricalLM (token embedding).

Both return (logits, state, aux) — no loss computation. The aux type A is
passed through from the backbone, making the wrapper backbone-agnostic.
"""

from __future__ import annotations

from typing import TypeVar, Generic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

A = TypeVar('A')


class LearnedNdEmbedding(nn.Module):
    """Factored learned positional embedding over N coordinate axes.

    Each sequence position maps to an N-dim coordinate via a lookup table,
    and the final embedding is the sum of per-axis learned embeddings.
    """

    def __init__(self, coords: Tensor, d_model: int):
        super().__init__()
        self.register_buffer('coords', coords)
        max_per_dim = coords.max(dim=0).values + 1
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(max_per_dim[i]), d_model)
            for i in range(coords.shape[1])
        ])

    def forward(self, positions: Tensor) -> Tensor:
        c = self.coords[positions]  # (T, N)
        return sum(emb(c[:, i]) for i, emb in enumerate(self.embeddings))


class PatchLM(nn.Module, Generic[A]):
    """LM wrapper with linear patch embedding.

    Input:  [B, T, patch_dim] float (normalized pixel values)
    Output: logits [B, T, patch_dim, vocab_size], state, aux
    """

    def __init__(
        self,
        backbone: nn.Module,
        embeddings: nn.Linear,
        lm_head: nn.Linear,
        wpe: nn.Module | None = None,
        patch_dim: int = 1,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.embeddings = embeddings
        self.lm_head = lm_head
        self.wpe = wpe
        self._patch_dim = patch_dim
        self._vocab_size = vocab_size

    def forward(self, idx: Tensor, state, train_step: tuple[int, int] | None = None):
        B, L, _PD = idx.shape
        backbone_state, offset = state
        positions = torch.arange(offset, offset + L, dtype=torch.long, device=idx.device)
        x = self.embeddings(idx)
        if self.wpe is not None:
            x = x + self.wpe(positions)
        x, backbone_state, aux = self.backbone(x, backbone_state, positions=positions, train_step=train_step)

        logits = self.lm_head(x)  # [B, T, patch_dim * vocab_size]
        logits = logits.view(B, L, self._patch_dim, self._vocab_size)

        new_state = (backbone_state, offset + L)
        return logits, new_state, aux

    def initial_state(self, batch_size):
        return (self.backbone.initial_state(batch_size), 0)

    def optim_groups(self) -> list[dict]:
        from utils import decay_nodecay_groups
        embed_params = list(self.embeddings.parameters()) + list(self.lm_head.parameters())
        if self.wpe is not None:
            embed_params += list(self.wpe.parameters())
        groups = decay_nodecay_groups(embed_params)
        groups += self.backbone.optim_groups()
        return groups

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
        """Generate patches. idx: [B, 1, patch_dim] float BOS."""
        cur = idx
        all_patches = []
        for _ in range(max_new_tokens):
            logits, state, _ = self(cur, state)  # [B, 1, patch_dim, vocab_size]
            logits = logits[:, -1] / temperature  # [B, patch_dim, vocab_size]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[..., [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            B, PD, V = probs.shape
            sampled = torch.multinomial(probs.view(B * PD, V), 1).view(B, PD)
            all_patches.append(sampled)
            cur = (sampled.float() / 255.0).unsqueeze(1)  # [B, 1, patch_dim]
        bos = torch.zeros(B, 1, self._patch_dim, dtype=torch.long, device=idx.device)
        return torch.cat([bos] + [p.unsqueeze(1) for p in all_patches], dim=1)  # [B, n+1, patch_dim]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and not getattr(module.weight, '_no_reinit', False):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1.0

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()


class CategoricalLM(nn.Module, Generic[A]):
    """LM wrapper with categorical (nn.Embedding) token embedding.

    Input:  [B, T] int64 token indices
    Output: logits [B, T, vocab_size], state, aux
    """

    def __init__(
        self,
        backbone: nn.Module,
        embeddings: nn.Embedding,
        lm_head: nn.Linear,
        wpe: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.embeddings = embeddings
        self.lm_head = lm_head
        self.wpe = wpe

    def forward(self, idx: Tensor, state, train_step: tuple[int, int] | None = None):
        B, L = idx.shape
        backbone_state, offset = state
        positions = torch.arange(offset, offset + L, dtype=torch.long, device=idx.device)
        x = self.embeddings(idx)
        if self.wpe is not None:
            x = x + self.wpe(positions)
        x, backbone_state, aux = self.backbone(x, backbone_state, positions=positions, train_step=train_step)

        logits = self.lm_head(x)  # [B, T, vocab_size]

        new_state = (backbone_state, offset + L)
        return logits, new_state, aux

    def initial_state(self, batch_size):
        return (self.backbone.initial_state(batch_size), 0)

    def optim_groups(self) -> list[dict]:
        from utils import decay_nodecay_groups
        embed_params = list(self.embeddings.parameters()) + list(self.lm_head.parameters())
        if self.wpe is not None:
            embed_params += list(self.wpe.parameters())
        groups = decay_nodecay_groups(embed_params)
        groups += self.backbone.optim_groups()
        return groups

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
        cur = idx
        for _ in range(max_new_tokens):
            logits, state, _ = self(cur, state)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            cur = idx_next
        return idx

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and not getattr(module.weight, '_no_reinit', False):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1.0

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()
