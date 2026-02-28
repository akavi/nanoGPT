"""Tests for Hilbert curve ordering used in face_hilbert_raster_config.py."""

import pytest
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Import the Hilbert helpers directly from the config.
# The config runs model init at import time which needs CUDA, so we replicate
# the pure-math functions here to test them in isolation.
# ---------------------------------------------------------------------------

def _hilbert_d2xy(n, d):
    """Convert Hilbert curve index d to (x, y) in an n x n grid."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if ((d & 1) ^ rx) else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def _build_hilbert_order(h, w):
    """Build permutation: hilbert_order[i] = linear index of the i-th Hilbert curve pixel."""
    n = max(h, w)
    assert n & (n - 1) == 0, "Hilbert curve requires power-of-2 grid"
    order = []
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if x < w and y < h:
            order.append(y * w + x)
    return order


# Precompute for the 32x32 case used in the config
H, W = 32, 32
HILBERT_ORDER = _build_hilbert_order(H, W)
INV_HILBERT_ORDER = [0] * len(HILBERT_ORDER)
for _i, _li in enumerate(HILBERT_ORDER):
    INV_HILBERT_ORDER[_li] = _i

HILBERT_ORDER_T = torch.tensor(HILBERT_ORDER, dtype=torch.long)
INV_HILBERT_ORDER_T = torch.tensor(INV_HILBERT_ORDER, dtype=torch.long)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHilbertD2XY:
    """Test the low-level d2xy conversion."""

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_bijection(self, n):
        """Every d in [0, n*n) maps to a unique (x, y) within the grid."""
        seen = set()
        for d in range(n * n):
            x, y = _hilbert_d2xy(n, d)
            assert 0 <= x < n, f"x={x} out of range for d={d}"
            assert 0 <= y < n, f"y={y} out of range for d={d}"
            assert (x, y) not in seen, f"duplicate (x,y)=({x},{y}) at d={d}"
            seen.add((x, y))
        assert len(seen) == n * n

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_adjacency(self, n):
        """Consecutive Hilbert indices must map to adjacent grid cells (Manhattan distance 1)."""
        prev = _hilbert_d2xy(n, 0)
        for d in range(1, n * n):
            cur = _hilbert_d2xy(n, d)
            dist = abs(cur[0] - prev[0]) + abs(cur[1] - prev[1])
            assert dist == 1, (
                f"n={n}, d={d}: ({prev[0]},{prev[1]}) -> ({cur[0]},{cur[1]}) "
                f"has Manhattan distance {dist}, expected 1"
            )
            prev = cur

    def test_known_n2(self):
        """Verify the n=2 Hilbert curve produces a valid U-shape."""
        coords = [_hilbert_d2xy(2, d) for d in range(4)]
        # This implementation uses a transposed orientation: (0,0) -> (0,1) -> (1,1) -> (1,0)
        assert coords == [(0, 0), (0, 1), (1, 1), (1, 0)]


class TestHilbertOrder:
    """Test the HILBERT_ORDER permutation and its inverse."""

    def test_is_permutation(self):
        """HILBERT_ORDER should be a permutation of [0, H*W)."""
        assert sorted(HILBERT_ORDER) == list(range(H * W))

    def test_inverse_is_correct(self):
        """INV_HILBERT_ORDER[HILBERT_ORDER[i]] == i for all i."""
        for i in range(H * W):
            assert INV_HILBERT_ORDER[HILBERT_ORDER[i]] == i

    def test_forward_inverse_identity(self):
        """Composing the permutation with its inverse yields identity."""
        idx = list(range(H * W))
        # forward then inverse
        permuted = [idx[HILBERT_ORDER[i]] for i in range(H * W)]
        recovered = [0] * (H * W)
        for i, li in enumerate(HILBERT_ORDER):
            recovered[li] = permuted[i]
        assert recovered == idx


class TestRoundtrip:
    """Test that tokenize (linear->hilbert) + detokenize (hilbert->linear) is identity."""

    def test_roundtrip_sequential(self):
        """A sequential pixel ramp [0..1023] should survive the roundtrip."""
        linear = torch.arange(H * W)
        # Tokenize: linear -> hilbert (as done in get_batch)
        hilbert = linear[HILBERT_ORDER_T]
        # Detokenize: hilbert -> linear (as done in detokenize)
        recovered = torch.zeros(H * W, dtype=linear.dtype)
        recovered[HILBERT_ORDER_T] = hilbert
        assert torch.equal(recovered, linear)

    def test_roundtrip_random(self):
        """Random pixel values should survive the roundtrip."""
        rng = torch.Generator().manual_seed(42)
        linear = torch.randint(0, 256, (H * W,), generator=rng)
        hilbert = linear[HILBERT_ORDER_T]
        recovered = torch.zeros(H * W, dtype=linear.dtype)
        recovered[HILBERT_ORDER_T] = hilbert
        assert torch.equal(recovered, linear)

    def test_roundtrip_batched(self):
        """Roundtrip should work on batched data (as in actual training)."""
        B = 8
        rng = torch.Generator().manual_seed(123)
        linear = torch.randint(0, 256, (B, H * W), generator=rng)
        # Tokenize (batch)
        hilbert = linear[:, HILBERT_ORDER_T]
        # Detokenize (batch)
        recovered = torch.zeros_like(linear)
        recovered[:, HILBERT_ORDER_T] = hilbert
        assert torch.equal(recovered, linear)

    def test_roundtrip_2d_image(self):
        """Roundtrip through actual 2D image reshape should preserve spatial content."""
        # Create a simple test image: gradient
        img = np.arange(H * W, dtype=np.uint8).reshape(H, W)
        linear = torch.from_numpy(img.reshape(-1))

        # Tokenize
        hilbert = linear[HILBERT_ORDER_T]

        # Detokenize (matches config's detokenize function)
        recovered_linear = torch.zeros(H * W, dtype=linear.dtype)
        recovered_linear[HILBERT_ORDER_T] = hilbert
        recovered_img = np.asarray(recovered_linear, dtype=np.uint8).reshape(H, W)

        np.testing.assert_array_equal(recovered_img, img)

    def test_inverse_order_roundtrip(self):
        """Using INV_HILBERT_ORDER should also roundtrip correctly."""
        linear = torch.arange(H * W)
        # Forward with HILBERT_ORDER
        hilbert = linear[HILBERT_ORDER_T]
        # Backward with INV_HILBERT_ORDER (gather-based)
        recovered = hilbert[INV_HILBERT_ORDER_T]
        assert torch.equal(recovered, linear)


class TestDetokenizeMatchesConfig:
    """Verify our detokenize matches the config's logic exactly."""

    def test_detokenize_with_bos(self):
        """Test detokenize strips BOS and unscrambles correctly."""
        BOS_ID = 0
        linear_pixels = torch.randint(0, 256, (H * W,))
        # Simulate what get_batch does: linear -> hilbert, prepend BOS
        hilbert_pixels = linear_pixels[HILBERT_ORDER_T]
        tokens = torch.cat([torch.tensor([BOS_ID]), hilbert_pixels])

        # Detokenize (replica of config's detokenize)
        t = tokens[1:]  # strip BOS
        recovered = torch.zeros(H * W, dtype=t.dtype)
        recovered[HILBERT_ORDER_T] = t
        img = np.asarray(recovered, dtype=np.uint8).reshape(H, W)

        expected = np.asarray(linear_pixels, dtype=np.uint8).reshape(H, W)
        np.testing.assert_array_equal(img, expected)
