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
from src.supervised.supervised_core import perform_split  # noqa: E402
from src.utils import FALLBACKS_USED, report_fallbacks_used

@click.command()
@click.option('--input-csv', type=click.Path(exists=True))
@click.option('--target-var', type=str, required=True, help="Column name to use as target (y)")
@click.option('--x-train-out', type=click.Path())
@click.option('--x-test-out', type=click.Path())
@click.option('--y-train-out', type=click.Path())
@click.option('--y-test-out', type=click.Path())
def main(input_csv, target_var, x_train_out, x_test_out, y_train_out, y_test_out):
    df = pd.read_csv(input_csv)
    X_train, X_test, y_train, y_test = perform_split(df, target_var)
    X_train.to_csv(x_train_out, index=False)
    X_test.to_csv(x_test_out, index=False)
    y_train.to_csv(y_train_out, index=False)
    y_test.to_csv(y_test_out, index=False)
    click.echo("Data split into train and test sets successfully.")
    report_fallbacks_used()


if __name__ == '__main__':
    main()
