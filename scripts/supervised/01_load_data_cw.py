import click
from pathlib import Path
import sys


# Ensure project root is on the Python path
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_duck_data  # noqa: E402


COLUMNS_TO_KEEP = [
    "passenger_injured",
    "day_of_year",
    "hour_of_day",
    "speed_limit",
    "engine_capacity_cc",
    "vehicle_age",
    "driver_age",
]


@click.command()
@click.option("--sql-file", type=click.Path(exists=True))
@click.option("--duck-file", type=click.Path(exists=True))
@click.option("--output-csv", type=click.Path())
def main(sql_file, duck_file, output_csv):
    """
    Load Stats19-derived data from Duck, clean it,
    and write a modelling-ready CSV.
    """
    with open(sql_file, "r") as f:
        query = f.read()

    df = load_duck_data(query, duck_file)
    df = df[COLUMNS_TO_KEEP]
    df_clean = (df.dropna().loc[~(df == -1).any(axis=1)].reset_index(drop=True))
    outpath = Path(outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_csv, index=False)
    click.echo(f"Rows before cleaning: {len(df)}")
    click.echo(f"Rows after cleaning : {len(df_clean)}")
    click.echo("")
    click.echo(df_clean.head())


if __name__ == "__main__":
    main()
