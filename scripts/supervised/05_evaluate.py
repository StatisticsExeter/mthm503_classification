import click
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# The next few lines ensure that the project root is in the Python path,
# allowing us to import modules from the project.
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[2]
if (str(PROJECT_ROOT) not in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
from src.supervised.supervised_core import (  # noqa: E402
    get_metrics_df, compute_roc_data, plot_roc, encode_labels)
from src.utils import FALLBACKS_USED, report_fallbacks_used

@click.command()
@click.option('--y-test-csv', type=click.Path(exists=True))
@click.option('--classifier-pred-csv', type=click.Path(exists=True))
@click.option('--classifier-prob-csv', type=click.Path(exists=True))
@click.option('--classifier-metrics-out', type=click.Path())
@click.option('--roc-plot-out', type=click.Path())
def main(y_test_csv, classifier_pred_csv, classifier_prob_csv,
         classifier_metrics_out, roc_plot_out):
    y_test = pd.read_csv(y_test_csv).iloc[:, 0]
    classifier_pred = pd.read_csv(classifier_pred_csv).iloc[:, 0]
    # Assuming column 1 contains the positive class probability
    classifier_prob = pd.read_csv(classifier_prob_csv).iloc[:, 1]
    get_metrics_df(y_test, classifier_pred).to_csv(classifier_metrics_out)
    y_labelled = encode_labels(y_test)
    classifier_roc = compute_roc_data(y_labelled, classifier_prob)
    fig = plot_roc(classifier_roc)
    fig.savefig(roc_plot_out, dpi=300)
    plt.close(fig)
    click.echo("Evaluation complete.\n")
    click.echo(f"Reports and plots saved in {classifier_metrics_out} and")
    click.echo(f"{roc_plot_out} respectively.")
    report_fallbacks_used()

if __name__ == '__main__':
    main()
