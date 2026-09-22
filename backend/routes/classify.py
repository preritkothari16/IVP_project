import io
import os
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Query
from ..core.preprocess import preprocess_image, to_uint8
from ..core.features import image_features, extract_features_from_folder
from ..core.classifier import evaluate

router = APIRouter(prefix="/api/classify", tags=["classify"])

_cached = {'feature_matrix': None, 'labels': None, 'image_data': None}


def _load_features(preprocessed_root: str):
    if _cached['feature_matrix'] is not None and _cached.get('root') == preprocessed_root:
        return _cached['feature_matrix'], _cached['labels'], _cached['image_data']
    result = extract_features_from_folder(preprocessed_root)
    _cached['feature_matrix'] = result['feature_matrix']
    _cached['labels'] = result['labels']
    _cached['image_data'] = result['image_data']
    _cached['root'] = preprocessed_root
    return result['feature_matrix'], result['labels'], result['image_data']


@router.get("/dataset-info")
def dataset_info(preprocessed_root: str = Query(...)):
    from collections import Counter
    class_counts = {}
    total = 0
    for name in sorted(os.listdir(preprocessed_root)):
        class_dir = os.path.join(preprocessed_root, name)
        if not os.path.isdir(class_dir):
            continue
        count = len([f for f in os.listdir(class_dir) if f.endswith('_preprocessed.png')])
        if count > 0:
            class_counts[name] = count
            total += count
    return {
        'num_classes': len(class_counts),
        'num_images': total,
        'num_features': 50,
        'class_counts': class_counts,
    }


@router.post("/evaluate")
def run_evaluation(
    preprocessed_root: str = Query(...),
    train_fraction: float = Query(0.75),
    k: int = Query(3),
):
    feat_matrix, labels, _ = _load_features(preprocessed_root)
    return evaluate(feat_matrix, labels, train_fraction, k)


@router.post("/single")
async def classify_single(
    file: UploadFile = File(...),
    preprocessed_root: str = Query(...),
    k: int = Query(3),
):
    from collections import Counter
    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert('RGB'))
    result = preprocess_image(img)
    masked = result['masked_output']
    mask = result['clean_mask'] > 0
    features = image_features(to_uint8(masked), mask)
    feat_matrix, labels, _ = _load_features(preprocessed_root)
    means = feat_matrix.mean(axis=0)
    stds = feat_matrix.std(axis=0)
    stds[stds == 0] = 1
    train_norm = (feat_matrix - means) / stds
    test_norm = (features - means) / stds
    dists = np.sum((train_norm - test_norm) ** 2, axis=1)
    neighbor_idx = np.argsort(dists)[:k]
    neighbor_labels = [labels[j] for j in neighbor_idx]
    prediction = Counter(neighbor_labels).most_common(1)[0][0]
    confidence = Counter(neighbor_labels).most_common(1)[0][1] / k
    return {
        'prediction': prediction,
        'confidence': confidence,
        'neighbors': neighbor_labels,
    }
