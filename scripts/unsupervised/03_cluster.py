import click
import pandas as pd
import sys
from pathlib import Path
# The next few lines ensure that the project root is in the Python path,
# allowing us to import modules from the project.
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[2]
if (str(PROJECT_ROOT) not in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
# Now we can import any project-specific utilities if needed
from src.unsupervised.unsupervised_core import (  # noqa: E402
    scale_data, compute_linkage, create_dendrogram, get_cluster_labels,
    project_pca, plot_cluster_scatter, run_kmeans, plot_centroids)
from src.utils import FALLBACKS_USED, report_fallbacks_used

@click.command()
@click.option("--input-csv", type=click.Path(exists=True))
@click.option("--out-dir", type=click.Path())
@click.option("--height", type=int, default=20, help="Height to cut the dendrogram")
@click.option("--k", type=int, default=4)
def main(input_csv, out_dir, height, k):
    df = pd.read_csv(input_csv)
    print(df.head())
    scaler, df_scaled = scale_data(df)
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    # Hierarchical Clustering
    Z = compute_linkage(df_scaled)
    fig_dendro = create_dendrogram(Z, title="Dendrogram")
    fig_dendro.savefig(p / "dendrogram.jpg", format='jpg', dpi=150)
    click.echo(f"Dendrogram saved in {p}")
    labels = get_cluster_labels(Z, height=height)
    df_pca = project_pca(df_scaled)
    fig = plot_cluster_scatter(df_pca, labels, title=f"Hierarchical Clusters (Height {height})")
    fig.savefig(p / "hcluster_scatter.jpg", dpi=150)
    click.echo(f"Scatter plot saved in {p}")
    # Now for kmeans
    model = run_kmeans(df_scaled, k)
    df_pca['cluster'] = model.labels_.astype(str)
    fig_scatter = plot_cluster_scatter(df_pca, model.labels_, k)
    figs = plot_centroids(model.cluster_centers_, scaler, df_scaled.columns, k)
    figs[0].savefig(p / "kmeans_1.jpg", dpi=150)
    figs[1].savefig(p / "kmeans_2.jpg", dpi=150)
    fig_scatter.savefig(p / "kmeans_scatter.jpg", dpi=150)
    click.echo(f"K-means centroid plots saved with prefix: {p}")
    report_fallbacks_used()


if __name__ == "__main__":
    main()
