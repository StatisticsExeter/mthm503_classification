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
    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(sql_file, 'r') as file:
        query = file.read()
        df = load_duck_data(query, duck_file)
    df.to_csv(p, index=False)
    print(df.head())


if __name__ == "__main__":
    main()
