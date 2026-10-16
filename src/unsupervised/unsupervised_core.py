import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from src.unsupervised.stubs import (
    stub_scale_data, stub_run_kmeans, stub_compute_linkage, stub_get_cluster_labels,
    stub_project_pca, stub_get_summary_stats)
from src.utils import NotImplementedExercise, fallback_to


@fallback_to(stub_scale_data)
def scale_data(df):
    """
    Instructions:
    1. Select only the numeric columns from the input DataFrame.
    2. Initialize and fit a StandardScaler to these columns.
    3. Transform the data and create a NEW DataFrame with the same column
       names and index as the original.
    4. Return a tuple containing: (the fitted scaler object, the scaled DataFrame).
    """
    raise NotImplementedExercise


@fallback_to(stub_run_kmeans)
def run_kmeans(data, k):
    """
    Instructions:
    1. Initialize a KMeans object with 'k' clusters and a random_state of 42.
    2. Fit the model to the provided 'data'.
    3. Return the fitted KMeans object.
    """
    raise NotImplementedExercise


@fallback_to(stub_compute_linkage)
def compute_linkage(scaled_data, method='ward'):
    """
    Instructions:
    1. Use the 'linkage' function from scipy.cluster.hierarchy.
    2. Compute the linkage matrix using the specified 'method' (default is 'ward').
    3. Return the linkage matrix (Z).
    """
    raise NotImplementedExercise


def create_dendrogram(Z, title="Hierarchical Clustering Dendrogram"):
    """
    Instructions:
    1. Create a Matplotlib figure and axis (figsize 10, 6).
    2. Use the 'dendrogram' function from scipy.cluster.hierarchy with the linkage matrix Z.
    3. Set the title, x-label, and y-label as specified.
    4. Return the Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Sample Index or Cluster Size')
    ax.set_ylabel('Distance (Ward)')
    fig.tight_layout()
    return fig


@fallback_to(stub_get_cluster_labels)
def get_cluster_labels(Z, height):
    """
    Instructions:
    1. Use the 'fcluster' function from scipy.cluster.hierarchy.
    2. Pass the linkage matrix 'Z' and the distance threshold 'height'.
    3. Use the criterion 'distance' to determine where to cut the dendrogram.
    4. Return the resulting array of cluster assignments.

    Parameters:
    Z (ndarray): The linkage matrix computed from compute_linkage.
    height (float): The distance threshold (y-axis on the dendrogram) at which to cut.

    Returns:
    ndarray: An array of cluster labels for each sample.
    """
    raise NotImplementedExercise


@fallback_to(stub_project_pca)
def project_pca(df, n_components=2):
    """
    Instructions:
    1. Initialize a PCA object from sklearn.decomposition with the specified
       number of components.
    2. Fit the PCA model to the input DataFrame and transform the data
       into the principal component space.
    3. Convert the resulting NumPy array into a pandas DataFrame.
    4. Name the columns 'PC1', 'PC2', etc., based on the number of components.
    5. Ensure the index of the new DataFrame matches the index of the input DataFrame.

    Parameters:
    df (pd.DataFrame): Input data containing only numeric, scaled variables.
    n_components (int): The number of principal components to calculate.

    Returns:
    pd.DataFrame: A DataFrame containing the projected Z-values (principal components).
    """
    raise NotImplementedExercise


def plot_cluster_scatter(df_pca, labels, title="Cluster Visualisation"):
    """
    Generates a 2D projection of the clustering results using Principal Components.

    This function visualizes high-dimensional clusters by plotting the first two
    principal components (PC1 and PC2). While the clustering itself may have
    happened across many variables, this plot 'squashes' that information into
    two dimensions to help us see if the clusters are spatially distinct.

    Args:
        df_pca (pd.DataFrame): A DataFrame containing at least two columns,
            'PC1' and 'PC2', representing the coordinates of the data points
            in the principal component space.
        labels (array-like): An array of integers or strings representing
            the cluster assignment for each row in df_pca. Used to color-code
            the points.
        title (str): The text to display at the top of the figure.

    Returns:
        matplotlib.figure.Figure: A figure object showing a colored scatter plot
            with a legend identifying each cluster.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], c=labels, cmap='viridis', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    # Add a legend
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    fig.tight_layout()
    return fig


def create_boxplot(df, title="Box Plot"):
    """Takes a DataFrame, returns a Matplotlib Figure."""
    df_numeric = df.select_dtypes(include=['number'])
    # We create the figure explicitly to return it
    fig, ax = plt.subplots(figsize=(10, 6))
    df_numeric.boxplot(ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Value')
    ax.set_xlabel('Variable')
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig


@fallback_to(stub_get_summary_stats)
def get_summary_stats(df):
    """Takes a DataFrame, returns a formatted string."""
    raise NotImplementedExercise


def _scatter(df, title):
    """
    Takes a dataframe 'df' and a string 'title'.
    Generates a scatter matrix of all numeric variables using Matplotlib.
    """
    # 1. Filter for only numeric columns to avoid errors
    df_numeric = df.select_dtypes(include=['number'])
    # 2. Create the scatter matrix
    # figsize ensures the plots aren't too cramped
    _ = pd.plotting.scatter_matrix(
        df_numeric,
        alpha=0.5,
        figsize=(12, 12),
        diagonal='hist'
    )
    # 3. Add the title
    # We use suptitle because scatter_matrix is a grid of subplots
    plt.suptitle(title, fontsize=16)
    # 4. Adjust layout so labels don't overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return plt.gcf()


def plot_centroids(scaled_centers, scaler, colnames, k):
    """
    Instructions:
    1. Create a DataFrame for 'scaled_centers' using the provided column names.
    2. Use the 'scaler' to inverse_transform the centers back to their original scale.
    3. Create a second DataFrame for these 'original_centers'.
    4. For each DataFrame (Scaled and Original), create a grouped bar chart.
    5. Each plot should show all features on the x-axis, with bars for each cluster.
    6. Return a list containing the two Figure objects.
    """
    # Prepare the two datasets
    original_centers = scaler.inverse_transform(scaled_centers)
    data_sets = [
        ("Original Scale", pd.DataFrame(original_centers, columns=colnames)),
        ("Standardized (Scaled) Scale", pd.DataFrame(scaled_centers, columns=colnames))
    ]
    figs = []
    for title_prefix, df in data_sets:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Setup for grouped bar chart
        x = np.arange(len(colnames))
        width = 0.8 / k
        for j in range(k):
            # Offset bars so they sit side-by-side for each cluster
            offset = (j - (k - 1) / 2) * width
            values = df.iloc[j].values
            ax.bar(x + offset, values, width, label=f'Cluster {j}')
        ax.set_ylabel('Value')
        ax.set_title(f'Cluster Centroids: {title_prefix}')
        ax.set_xticks(x)
        ax.set_xticklabels(colnames, rotation=45)
        ax.legend()
        fig.tight_layout()
        figs.append(fig)
    return figs
