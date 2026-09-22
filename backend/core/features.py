import os
import numpy as np
from PIL import Image
from .preprocess import to_uint8, region_props, rgb_to_hsv_vectorized


def glcm_features(gray_img: np.ndarray) -> np.ndarray:
    levels = 64
    max_val = gray_img.max()
    if max_val == 0:
        return np.array([0.0, 0.0, 1.0, 1.0])
    quantized = np.minimum(levels - 1, (gray_img / max_val * (levels - 1)).astype(np.int32))
    offsets = [(0, 1), (1, 1), (1, 0), (1, -1)]
    rows, cols = quantized.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for dr, dc in offsets:
        a = quantized[:rows - abs(dr), :cols - abs(dc)].ravel()
        b = quantized[max(0, dr):rows + min(0, dr), max(0, dc):cols + min(0, dc)].ravel()
        flat = a * levels + b
        counts = np.bincount(flat, minlength=levels * levels).reshape(levels, levels)
        glcm += counts + counts.T
    glcm = glcm / glcm.sum()
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
    contrast = np.sum((i_idx - j_idx) ** 2 * glcm)
    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    sigma_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * glcm))
    sigma_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * glcm))
    if sigma_i == 0 or sigma_j == 0:
        correlation = 0.0
    else:
        correlation = np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm) / (sigma_i * sigma_j)
    energy = np.sum(glcm ** 2)
    homogeneity = np.sum(glcm / (1 + np.abs(i_idx - j_idx)))
    return np.array([contrast, correlation, energy, homogeneity])


def skewness(values: np.ndarray) -> float:
    values = values.astype(np.float64).ravel()
    deviation = values - values.mean()
    sigma = np.sqrt(np.mean(deviation ** 2))
    if sigma == 0:
        return 0.0
    return float(np.mean(deviation ** 3) / sigma ** 3)


def normalized_histogram(channel: np.ndarray, mask: np.ndarray, bin_count: int) -> np.ndarray:
    edges = np.linspace(0, 1, bin_count + 1)
    pixels = channel[mask].ravel()
    hist, _ = np.histogram(pixels, bins=edges)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def image_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(50)
    hsv = rgb_to_hsv_vectorized(img)
    h_hist = normalized_histogram(hsv[:, :, 0], mask, 16)
    s_hist = normalized_histogram(hsv[:, :, 1], mask, 8)
    v_hist = normalized_histogram(hsv[:, :, 2], mask, 4)
    moments = np.zeros(9)
    for ch in range(3):
        values = img[:, :, ch].astype(np.float64)
        vals = values[mask]
        idx = ch * 3
        moments[idx] = vals.mean()
        moments[idx + 1] = vals.std()
        moments[idx + 2] = skewness(vals)
    gray = 0.2989 * img[:, :, 0].astype(np.float64) + \
           0.5870 * img[:, :, 1].astype(np.float64) + \
           0.1140 * img[:, :, 2].astype(np.float64)
    texture = glcm_features(gray)
    props = region_props(mask)
    shape = np.array([
        props['Area'], props['Perimeter'], props['CentroidX'], props['CentroidY'],
        props['MajorAxisLength'], props['MinorAxisLength'],
        props['Eccentricity'], props['Solidity'], props['Extent'],
    ])
    return np.concatenate([h_hist, s_hist, v_hist, moments, texture, shape])


_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')


def extract_features_from_folder(preprocessed_root: str) -> dict:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_key = preprocessed_root.replace('\\', '_').replace(':', '').replace('/', '_')
    cache_file = os.path.join(_CACHE_DIR, f'{cache_key}.npz')

    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        feature_matrix = data['features']
        labels = data['labels'].tolist()
        paths = data['paths'].tolist()
        image_data = [{'path': p, 'class': l, 'filename': os.path.basename(p), 'features': feature_matrix[i]}
                      for i, (p, l) in enumerate(zip(paths, labels))]
        return {
            'feature_matrix': feature_matrix,
            'labels': labels,
            'feature_names': [f'Feature{i:03d}' for i in range(1, feature_matrix.shape[1] + 1)],
            'image_data': image_data,
            'num_images': len(labels),
            'num_features': feature_matrix.shape[1],
        }

    image_data = []
    for class_name in sorted(os.listdir(preprocessed_root)):
        class_dir = os.path.join(preprocessed_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.endswith('_preprocessed.png'):
                continue
            img_path = os.path.join(class_dir, fname)
            mask_path = img_path.replace('_preprocessed.png', '_mask.png')
            img = to_uint8(np.array(Image.open(img_path).convert('RGB')))
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert('L')) > 0
            else:
                mask = np.any(img > 0, axis=2)
            features = image_features(img, mask)
            image_data.append({
                'path': img_path,
                'class': class_name,
                'filename': fname,
                'features': features,
            })

    feature_matrix = np.array([d['features'] for d in image_data])
    labels = [d['class'] for d in image_data]
    paths = [d['path'] for d in image_data]
    np.savez_compressed(cache_file, features=feature_matrix, labels=labels, paths=paths)

    return {
        'feature_matrix': feature_matrix,
        'labels': labels,
        'feature_names': [f'Feature{i:03d}' for i in range(1, 51)],
        'image_data': image_data,
        'num_images': len(image_data),
        'num_features': 50,
    }
