import click
from pathlib import Path
import sys
# The next few lines ensure that the project root is in the Python path,
# allowing us to import modules from the project.
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[2]
if (str(PROJECT_ROOT) not in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
# Now we can import any project-specific utilities if needed
from src.utils import load_duck_data  # noqa: E402

COLUMNS_TO_KEEP = [
    "median_co2_emissions_current",
    "median_energy_consumption_current",
    "median_heating_cost_current",
    "median_hot_water_cost_current",
    "median_lighting_cost_current",
]


@click.command()
@click.option("--sql-file", type=click.Path(exists=True))
@click.option("--duck-file", type=click.Path(exists=True))
@click.option("--output-csv", type=click.Path())
def main(sql_file, output_csv):
    with open(sql_file, 'r') as file:
        query = file.read()
        df = load_duck_data(query, duck_file)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    missing = set(COLUMNS_TO_KEEP) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in query result: {sorted(missing)}"
        )
    df_clean = df[COLUMNS_TO_KEEP].copy()
    df_clean = df_clean.dropna().reset_index(drop=True)
    df_clean.to_csv(output_csv, index=False)
    click.echo(f"Data saved to {output_csv}")
    click.echo(f"Rows written: {len(df_clean)}")
    click.echo("Columns used:")
    for col in df_clean.columns:
        click.echo(f"  - {col}")
    click.echo("")
    click.echo(df_clean.head())


if __name__ == "__main__":
    main()
