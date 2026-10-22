import click
import pandas as pd
from pathlib import Path
import sys
# The next few lines ensure that the project root is in the Python path,
# allowing us to import modules from the project.
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[2]
if (str(PROJECT_ROOT) not in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
from src.supervised.supervised_core import scatter_by_group

@click.command()
@click.option("--input-csv", type=click.Path(exists=True), required=True,
              help="Input CSV file.")
@click.option("--target-var", required=True,
              help="Name of the grouping/target variable.")
@click.option("--plot-out", type=click.Path(), required=True,
              help="Path to save the plot.")
@click.option("--title", required=True, help="Plot title.")
def main(input_csv, target_var, plot_out, title):
    """
    Generate scatterplots of all pairs of continuous variables,
    coloured and shaped by the target variable.
    """
    df = pd.read_csv(input_csv)
    print(df.head())
    print(df.columns)
    fig = scatter_by_group(df=df, group_var=target_var, title=title)
    fig.savefig(plot_out, bbox_inches="tight", dpi=300)
    print(f"Saved plot to {plot_out}")


if __name__ == "__main__":
    main()
