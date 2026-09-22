import numpy as np
from collections import Counter


def knn_classify(train_x: np.ndarray, train_y: list[str],
                 test_x: np.ndarray, k: int = 3) -> list[str]:
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1
    train_norm = (train_x - means) / stds
    test_norm = (test_x - means) / stds
    predictions = []
    for i in range(test_norm.shape[0]):
        dists = np.sum((train_norm - test_norm[i]) ** 2, axis=1)
        neighbor_idx = np.argsort(dists)[:k]
        neighbor_labels = [train_y[j] for j in neighbor_idx]
        counts = Counter(neighbor_labels)
        predictions.append(counts.most_common(1)[0][0])
    return predictions


def stratified_split(labels: list[str], train_fraction: float = 0.75,
                     seed: int = 1) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    classes = list(set(labels))
    train_idx = []
    test_idx = []
    for cls in classes:
        indices = [i for i, l in enumerate(labels) if l == cls]
        rng.shuffle(indices)
        n_train = max(1, min(len(indices) - 1, int(train_fraction * len(indices))))
        train_idx.extend(indices[:n_train])
        test_idx.extend(indices[n_train:])
    return train_idx, test_idx


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict:
    classes = sorted(set(y_true) | set(y_pred))
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for actual, predicted in zip(y_true, y_pred):
        matrix[class_to_idx[actual], class_to_idx[predicted]] += 1
    return {'classes': classes, 'matrix': matrix.tolist()}


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> list[dict]:
    classes = sorted(set(y_true) | set(y_pred))
    results = []
    for cls in classes:
        tp = sum(1 for a, p in zip(y_true, y_pred) if a == cls and p == cls)
        fp = sum(1 for a, p in zip(y_true, y_pred) if a != cls and p == cls)
        fn = sum(1 for a, p in zip(y_true, y_pred) if a == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for a in y_true if a == cls)
        results.append({
            'class': cls, 'precision': precision, 'recall': recall,
            'f1': f1, 'support': support,
        })
    return results


def evaluate(feature_matrix: np.ndarray, labels: list[str],
             train_fraction: float = 0.75, k: int = 3) -> dict:
    train_idx, test_idx = stratified_split(labels, train_fraction)
    train_x = feature_matrix[train_idx]
    test_x = feature_matrix[test_idx]
    train_y = [labels[i] for i in train_idx]
    test_y = [labels[i] for i in test_idx]
    predictions = knn_classify(train_x, train_y, test_x, k)
    accuracy = sum(1 for a, p in zip(test_y, predictions) if a == p) / len(test_y)
    cm = confusion_matrix(test_y, predictions)
    metrics = per_class_metrics(test_y, predictions)
    return {
        'accuracy': accuracy,
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'confusion_matrix': cm,
        'per_class_metrics': metrics,
        'predictions': predictions,
        'actual': test_y,
    }
