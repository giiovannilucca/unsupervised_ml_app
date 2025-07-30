import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Union, Dict

def get_metrics(X: np.ndarray, labels: np.ndarray) -> Union[Dict[str, float], None]:
    """
    Evaluates the quality of clustering using three internal validation metrics:
    - Silhouette Score
    - Calinski-Harabasz Index
    - Davies-Bouldin Index

    The function returns None if fewer than 2 valid clusters are found,
    which typically indicates poor clustering (e.g., all points are labeled as noise).

    Args:
        X (np.ndarray): Scaled feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Cluster labels assigned to each sample.

    Returns:
        dict[str, float] or None: Dictionary with metric names and their corresponding scores,
                                  or None if the metrics cannot be computed.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return None  # Métricas não são definidas com menos de 2 clusters válidos

    results = {
        "Silhouette Score": silhouette_score(X, labels),
        "Calinski-Harabasz Index": calinski_harabasz_score(X, labels),
        "Davies-Bouldin Index": davies_bouldin_score(X, labels)
    }
    return results