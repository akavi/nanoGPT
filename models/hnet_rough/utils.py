# ---------------------------------------------------------------------------
# x_pack_kwargs accessors — work with both the legacy dict format
# ({"lens_cs": ..., "max": ..., "batch": ..., "idx": ...}) and the new
# Tensor format (shape (1, N) of per-sequence lengths).
# ---------------------------------------------------------------------------

def get_lens_cs(x_pack_kwargs) -> torch.Tensor:
    """Cumulative sequence lengths (N+1,) with leading 0."""
    if isinstance(x_pack_kwargs, torch.Tensor):
        return F.pad(x_pack_kwargs[0].cumsum(0), (1, 0)).long()
    return x_pack_kwargs["lens_cs"]


def get_seq_lens(x_pack_kwargs) -> torch.Tensor:
    """Per-sequence lengths (N,)."""
    if isinstance(x_pack_kwargs, torch.Tensor):
        return x_pack_kwargs[0]
    lens_cs = x_pack_kwargs["lens_cs"]
    return lens_cs[1:] - lens_cs[:-1]


def get_max_seqlen(x_pack_kwargs) -> int:
    """Maximum sequence length across the batch."""
    if isinstance(x_pack_kwargs, torch.Tensor):
        return int(x_pack_kwargs[0].max().item())
    return x_pack_kwargs["max"]


def get_batch_size(x_pack_kwargs) -> int:
    """Original (pre-packing) batch size."""
    if isinstance(x_pack_kwargs, torch.Tensor):
        return 1
    return x_pack_kwargs["batch"]
