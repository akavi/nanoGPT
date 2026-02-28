"""Visualize Hilbert curve block boundaries overlaid on output images."""

import numpy as np
from PIL import Image, ImageDraw
import os

H, W = 32, 32

def _hilbert_d2xy(n, d):
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


def make_boundary_mask(block_size):
    """Create a mask showing boundaries between Hilbert sub-blocks of given size.

    A boundary pixel is one where consecutive Hilbert indices d and d+1
    cross a block_size boundary (i.e., d // block_size != (d+1) // block_size).
    We mark both pixels involved in such a transition.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    for d in range(H * W - 1):
        if d // block_size != (d + 1) // block_size:
            x0, y0 = _hilbert_d2xy(H, d)
            x1, y1 = _hilbert_d2xy(H, d + 1)
            mask[y0, x0] = 255
            mask[y1, x1] = 255
    return mask


def make_hilbert_curve_image():
    """Draw the Hilbert curve path on a blank image."""
    scale = 8  # pixels per grid cell
    img = Image.new("RGB", (W * scale, H * scale), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    points = []
    for d in range(H * W):
        x, y = _hilbert_d2xy(H, d)
        points.append((x * scale + scale // 2, y * scale + scale // 2))

    draw.line(points, fill=(0, 0, 0), width=1)

    # Mark block boundaries with colored dots
    colors = {4: (255, 0, 0), 16: (0, 255, 0), 64: (0, 0, 255), 256: (255, 165, 0)}
    for block_size, color in colors.items():
        for d in range(block_size, H * W, block_size):
            x, y = _hilbert_d2xy(H, d)
            cx, cy = x * scale + scale // 2, y * scale + scale // 2
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=color)

    return img


def overlay_boundaries_on_sample(sample_path, block_size):
    """Overlay Hilbert block boundaries on a sample image."""
    img = Image.open(sample_path).convert("RGB")
    img = img.resize((W * 8, H * 8), Image.NEAREST)  # scale up for visibility

    mask = make_boundary_mask(block_size)

    # Draw boundary pixels as red dots on the upscaled image
    draw = ImageDraw.Draw(img)
    scale = 8
    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                cx, cy = x * scale + scale // 2, y * scale + scale // 2
                draw.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 0, 0))

    return img


if __name__ == "__main__":
    out_dir = "tests/hilbert_viz"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Boundary masks at different scales
    for block_size in [4, 16, 64, 256]:
        mask = make_boundary_mask(block_size)
        Image.fromarray(mask, mode="L").save(f"{out_dir}/boundary_mask_{block_size}.png")

    # 2. Hilbert curve with boundary markers
    curve_img = make_hilbert_curve_image()
    curve_img.save(f"{out_dir}/hilbert_curve.png")

    # 3. Overlay boundaries on actual output samples
    sample_dir = "outputs/53"
    if os.path.exists(sample_dir):
        for block_size in [4, 16, 64, 256]:
            img = overlay_boundaries_on_sample(f"{sample_dir}/0.png", block_size)
            img.save(f"{out_dir}/sample0_boundaries_{block_size}.png")

    print(f"Saved visualizations to {out_dir}/")
