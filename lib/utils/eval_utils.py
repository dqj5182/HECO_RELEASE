import numpy as np


def compute_contact_metrics(probability, target):
    target = np.asarray(target) >= .5
    predicted = np.asarray(probability) >= .5
    positive = target.sum(1) > 0

    tp = (predicted & target).sum(1)[positive]
    predicted_count = predicted.sum(1)[positive]
    gt_count = target.sum(1)[positive]

    precision = np.divide(tp, predicted_count, out=np.zeros(len(tp), dtype=np.float64), where=predicted_count > 0)
    recall = tp / gt_count

    precision = float(precision.mean()) if len(tp) else 0.0
    recall = float(recall.mean()) if len(tp) else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'samples': int(len(target)), 'positive_samples': int(positive.sum()), 'threshold': .5}