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
from src.unsupervised.unsupervised_core import _scatter  # noqa: E402
from src.utils import FALLBACKS_USED, report_fallbacks_used

@click.command()
@click.option("--input-csv", type=click.Path(exists=True))
@click.option("--out-path", type=click.Path())
@click.option("--title", type=str)
def main(input_csv, out_path, title):
    out_path = Path(out_path)
    df = pd.read_csv(input_csv)
    fig = _scatter(df, title=title)
    fig.savefig(out_path, format='jpg', dpi=150)
    report_fallbacks_used()


if __name__ == "__main__":
    main()
