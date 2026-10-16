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


@click.command()
@click.option("--sql-file", type=click.Path(exists=True))
@click.option("--duck-file", type=click.Path(exists=True))
@click.option("--output-csv", type=click.Path())
def main(sql_file, duck_file, output_csv):
    query = Path(sql_file).read_text()
    df = load_duck_data(query, duck_file)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    print("AVAILABLE COLUMNS:")
    print(df.columns)
    TARGET_COLUMNS = ["local_authority_ons_district",
                      "n_raining", "n_dark", "n_dry", "n_urban", "n_slight"]
    df = df[TARGET_COLUMNS]
    print("SELECTED COLUMNS:")
    print(df.columns)
    print("THIS SELECTION IS CONTROLLED IN scripts/unsupervised/01_load_data_lab.py")
    df.to_csv(output_csv, index=False)
    click.echo(f"Data saved to {output_csv}")


if __name__ == "__main__":
    main()
