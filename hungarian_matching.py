######
# Modified version of cluster_map from: https://github.com/vlfom/RNCDL/blob/main/discovery/evaluation/evaluator_discovery.py
######

import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_map_fn(y_true, y_pred, last_free_class_id, max_class_num):
    """
    Args:
        last_free_class_id: least possible free ID to use for novel classes, defaults to 1203 as LVIS class IDs
                            range up to 1203
    """

    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    # Generate weight matrix
    max_y_pred = y_pred.max()
    max_y_true = y_true.max()

    w = np.zeros((max_y_pred + 1, max_y_true + 1), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Match
    x, y = linear_sum_assignment(w, maximize=True)
    mapping = list(zip(x, y))

    # Create fake extra classes for unmapped categories
    for i in range(0, max_class_num):
        if i not in x:
            mapping.append((i, last_free_class_id))
            last_free_class_id += 1

    return mapping
