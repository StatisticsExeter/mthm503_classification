import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage


def stub_scale_data(df):
    """
    Returns the numeric columns unchanged.
    Mimics the interface of scale_data().
    """
    print("USING STUB SCALE DATA")

    class StubScaler:
        def inverse_transform(self, X):
            return X
    numeric_cols = df.select_dtypes(include=["number"]).columns
    return StubScaler(), df[numeric_cols].copy()


def stub_run_kmeans(data, k):
    """
    Returns an object with a labels_ attribute.
    All observations belong to a single cluster.
    """
    class StubModel:
        def __init__(self, n):
            self.labels_ = np.repeat(1, n)
            n_features = data.shape[1]
            self.cluster_centers_ = np.zeros((k, n_features))
    return StubModel(len(data))


def stub_compute_linkage(scaled_data, method="ward"):
    """
    Produces a minimal valid linkage matrix.
    """
    if len(scaled_data) < 2:
        scaled_data = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    return linkage(scaled_data.iloc[:, :2], method="single")


def stub_get_cluster_labels(Z, height):
    """
    Places every observation in cluster 1.
    """
    n = Z.shape[0] + 1
    return np.repeat(1, n)


def stub_project_pca(df, n_components=2):
    """
    Creates a simple two-dimensional projection.
    """
    return pd.DataFrame({"PC1": np.arange(len(df)),
                         "PC2": np.zeros(len(df))},
                        index=df.index)


def stub_get_summary_stats(df):
    """
    Returns placeholder summary text.
    """
    return "Summary statistics not yet implemented."
