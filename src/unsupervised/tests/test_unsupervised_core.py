import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from src.unsupervised.unsupervised_core import (
    scale_data, run_kmeans, compute_linkage, plot_cluster_scatter, get_cluster_labels,
    project_pca)


def test_project_pca_dimensions():
    # 1. Create a dummy dataset with 4 features
    data = np.random.rand(10, 4)
    df = pd.DataFrame(data, columns=['feat1', 'feat2', 'feat3', 'feat4'])
    # 2. Run PCA to project down to 2 components
    result = project_pca(df, n_components=2)
    # 3. Assertions
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (10, 2), "Output should have 10 rows and 2 columns."
    assert list(result.columns) == ['PC1', 'PC2']
    # Ensure the index is preserved
    assert all(result.index == df.index)


def test_project_pca_variance():
    # Create data where PC1 should clearly capture most variance
    # x and y are highly correlated, z is random noise
    x = np.linspace(0, 10, 50)
    df = pd.DataFrame({
        'var1': x,
        'var2': x + np.random.normal(0, 0.1, 50),
        'var3': np.random.normal(0, 5, 50)
    })
    result = project_pca(df, n_components=2)
    # PC1 should have a higher standard deviation than PC2
    # (since it captures the most variance)
    assert result['PC1'].std() > result['PC2'].std()


def test_scale_data_integrity():
    df = pd.DataFrame({
        'A': [10, 20, 30],
        'B': [1, 2, 3],
        'C': ['text', 'only', 'here']
    })
    scaler, scaled_df = scale_data(df)
    # 1. Check types
    assert isinstance(scaler, StandardScaler)
    assert isinstance(scaled_df, pd.DataFrame)
    # 2. Check column selection
    assert list(scaled_df.columns) == ['A', 'B']
    # 3. Check math (Mean should be 0, Std should be 1)
    assert scaled_df['A'].mean() == pytest.approx(0)
    assert scaled_df['A'].std() == pytest.approx(1.225, rel=1e-2)  # Note: Std depends on ddof
    # 4. Check index preservation
    assert all(scaled_df.index == df.index)


def test_run_kmeans_labels():
    data = np.array([[1, 2], [1, 3], [10, 12], [10, 13]])
    k = 2
    model = run_kmeans(data, k)
    assert isinstance(model, KMeans)
    assert len(np.unique(model.labels_)) == k
    # Check if the points are grouped logically
    assert model.labels_[0] == model.labels_[1]
    assert model.labels_[2] == model.labels_[3]


def test_compute_linkage_shape():
    # Simple dataset: 4 points
    data = np.array([[1, 1], [1.1, 1.1], [5, 5], [10, 10]])
    Z = compute_linkage(data)
    # In SciPy linkage, for N samples, the matrix Z has shape (N-1, 4)
    assert Z.shape == (3, 4)
    # The first two points (0 and 1) should be merged first because they are closest
    # Z[0, 0] and Z[0, 1] are the indices of the elements merged in the first step
    assert int(Z[0, 0]) == 0
    assert int(Z[0, 1]) == 1


def test_plot_cluster_scatter_elements():
    # 1. Setup dummy PCA data and labels
    df_pca = pd.DataFrame({
        'PC1': [1, 2, 5, 6],
        'PC2': [1, 1, 5, 5]
    })
    labels = [0, 0, 1, 1]
    fig = plot_cluster_scatter(df_pca, labels)
    # 2. Verify it's a figure
    assert isinstance(fig, plt.Figure)
    # 3. Check for specific plot elements
    ax = fig.get_axes()[0]
    # Check labels
    assert ax.get_xlabel() == 'Principal Component 1'
    assert ax.get_ylabel() == 'Principal Component 2'
    # Check that a legend was actually added
    assert ax.get_legend() is not None
    assert ax.get_legend().get_title().get_text() == "Clusters"
    plt.close(fig)


def test_get_cluster_labels_threshold():
    # 1. Create a dataset with two very clear, distant groups
    # Group 1: near 0, Group 2: near 10
    data = np.array([[1, 1], [1.1, 1.1], [5, 5], [10, 10]])
    Z = linkage(data, method='ward')
    print(Z)
    # 2. Test a "Low" cut (should give 3 cluster)
    # The distance between the two main groups will be very large
    labels_low = get_cluster_labels(Z, height=1)
    assert len(np.unique(labels_low)) == 3
    # 3. Test a "high" cut (should give 1 clusters)
    labels_high = get_cluster_labels(Z, height=14)
    assert len(np.unique(labels_high)) == 1
    # 4. Verify output length matches input data length
    assert len(labels_low) == 4
