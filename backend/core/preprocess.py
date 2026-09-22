import numpy as np
from scipy import ndimage
from PIL import Image


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    values = img.astype(np.float64)
    vmin, vmax = values.min(), values.max()
    if vmin >= 0 and vmax <= 1:
        return np.clip(np.round(values * 255), 0, 255).astype(np.uint8)
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def contrast_stretch(channel: np.ndarray) -> np.ndarray:
    low, high = channel.min(), channel.max()
    if high > low:
        return (channel - low) / (high - low)
    return channel.copy()


def gaussian_filter(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel1d = kernel1d / kernel1d.sum()
    kernel2d = np.outer(kernel1d, kernel1d)
    channels = img.shape[2] if img.ndim == 3 else 1
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(channels):
        ch = img[:, :, c] if channels > 1 else img[:, :, 0]
        filtered = ndimage.convolve(ch.astype(np.float64), kernel2d, mode='constant')
        out[:, :, c] = np.clip(np.round(filtered), 0, 255).astype(np.uint8)
    return out


def rgb_to_hsv_vectorized(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    s = np.zeros_like(delta)
    nonzero = maxc > 0
    s[nonzero] = delta[nonzero] / maxc[nonzero]
    h = np.zeros_like(r)
    mask_r = (maxc == r) & (delta > 0)
    mask_g = (maxc == g) & (delta > 0)
    mask_b = (maxc == b) & (delta > 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360
    h = h / 360.0
    hsv = np.stack([h, s, v], axis=2)
    return hsv


def hsv_to_rgb_vectorized(hsv: np.ndarray) -> np.ndarray:
    h = hsv[:, :, 0] * 360.0
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    for lo, hi, ri, gi, bi in [
        (0, 60, 'c', 'x', 0), (60, 120, 'x', 'c', 0), (120, 180, 0, 'c', 'x'),
        (180, 240, 0, 'x', 'c'), (240, 300, 'x', 0, 'c'), (300, 360, 'c', 0, 'x'),
    ]:
        mask = (h >= lo) & (h < hi)
        vals = {'c': c, 'x': x, 0: np.zeros_like(c)}
        r[mask] = vals[ri][mask]
        g[mask] = vals[gi][mask]
        b[mask] = vals[bi][mask]
    rgb = np.stack([r + m, g + m, b + m], axis=2)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def connected_components(mask: np.ndarray):
    labels, num = ndimage.label(mask.astype(np.int32), structure=np.ones((3, 3), dtype=int))
    sizes = ndimage.sum(mask.astype(np.int32), labels, range(1, num + 1))
    return labels, np.array(sizes, dtype=np.int64)


def area_open(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, sizes = connected_components(mask)
    cleaned = np.zeros_like(mask)
    for i, size in enumerate(sizes):
        if size >= min_pixels:
            cleaned[labels == (i + 1)] = True
    return cleaned


def fill_holes(mask: np.ndarray) -> np.ndarray:
    rows, cols = mask.shape
    outside = np.zeros((rows, cols), dtype=bool)
    queue = []
    for r in range(rows):
        for c in [0, cols - 1]:
            if not mask[r, c] and not outside[r, c]:
                queue.append((r, c))
                outside[r, c] = True
    for c in range(cols):
        for r in [0, rows - 1]:
            if not mask[r, c] and not outside[r, c]:
                queue.append((r, c))
                outside[r, c] = True
    head = 0
    while head < len(queue):
        cr, cc = queue[head]
        head += 1
        for dr in [-1, 1]:
            nr = cr + dr
            if 0 <= nr < rows and not mask[nr, cc] and not outside[nr, cc]:
                queue.append((nr, cc))
                outside[nr, cc] = True
            nc = cc + dr
            if 0 <= nc < cols and not mask[cr, nc] and not outside[cr, nc]:
                queue.append((cr, nc))
                outside[cr, nc] = True
    return mask | ~outside


def morph_close(mask: np.ndarray, radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    se = (x ** 2 + y ** 2) <= radius ** 2
    se_sum = se.sum()
    dilated = ndimage.convolve(mask.astype(np.float64), se.astype(np.float64), mode='constant') > 0
    closed = ndimage.convolve(dilated.astype(np.float64), se.astype(np.float64), mode='constant') == se_sum
    return closed


def region_props(mask: np.ndarray) -> dict:
    labels, sizes = connected_components(mask)
    names = ['Area', 'Perimeter', 'CentroidX', 'CentroidY', 'MajorAxisLength',
             'MinorAxisLength', 'Eccentricity', 'Solidity', 'Extent']
    zeros = {k: 0.0 for k in names}
    if sizes.size == 0:
        return zeros
    largest = np.argmax(sizes)
    component = labels == (largest + 1)
    rows, cols = np.where(component)
    area = len(rows)
    centroid_x = float(cols.mean())
    centroid_y = float(rows.mean())
    kernel = np.ones((3, 3), dtype=np.float64)
    neighbor_count = ndimage.convolve(component.astype(np.float64), kernel, mode='constant')
    boundary = component & (neighbor_count < 9)
    perimeter = float(boundary.sum())
    centered = np.column_stack([cols - centroid_x, rows - centroid_y])
    cov = (centered.T @ centered) / area
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = 4 * np.sqrt(max(eigvals[0], 0))
    minor = 4 * np.sqrt(max(eigvals[1], 0)) if len(eigvals) > 1 else 0
    eccentricity = float(np.sqrt(max(0, 1 - (minor / major) ** 2))) if major > 0 else 0.0
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    extent = area / max((max_r - min_r + 1) * (max_c - min_c + 1), 1)
    if area >= 3:
        from scipy.spatial import ConvexHull
        points = np.column_stack([cols, rows])
        hull = ConvexHull(points)
        hull_area = hull.volume
        solidity = min(1.0, area / max(hull_area, 1))
    else:
        solidity = 1.0
    return {
        'Area': float(area), 'Perimeter': perimeter,
        'CentroidX': centroid_x, 'CentroidY': centroid_y,
        'MajorAxisLength': float(major), 'MinorAxisLength': float(minor),
        'Eccentricity': eccentricity, 'Solidity': float(solidity),
        'Extent': float(extent),
    }


def preprocess_image(img: np.ndarray, target_size=(256, 256)) -> dict:
    original = to_uint8(img)
    pil_img = Image.fromarray(original)
    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    resized = np.array(pil_img)

    filtered = gaussian_filter(resized, 1.5)

    hsv_filtered = rgb_to_hsv_vectorized(filtered)
    hsv_filtered[:, :, 2] = contrast_stretch(hsv_filtered[:, :, 2])
    enhanced = hsv_to_rgb_vectorized(hsv_filtered)

    enhanced_hsv = rgb_to_hsv_vectorized(enhanced)
    raw_mask = (
        (enhanced_hsv[:, :, 0] >= 0.05) & (enhanced_hsv[:, :, 0] <= 0.45) &
        (enhanced_hsv[:, :, 1] >= 0.15) & (enhanced_hsv[:, :, 2] >= 0.10)
    )
    area_mask = area_open(raw_mask, 500)
    clean_mask = morph_close(fill_holes(area_mask), 3)
    masked_output = enhanced * clean_mask[:, :, np.newaxis].astype(np.uint8)

    return {
        'original': original,
        'resized': resized,
        'filtered': filtered,
        'enhanced': enhanced,
        'clean_mask': clean_mask.astype(np.uint8) * 255,
        'masked_output': masked_output,
    }
