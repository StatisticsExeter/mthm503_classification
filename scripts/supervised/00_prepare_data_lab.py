import click
from pathlib import Path
import sys
import duckdb
# The next few lines ensure that the project root is in the Python path,
# allowing us to import modules from the project.
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[2]
if (str(PROJECT_ROOT) not in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
# Now we can import any project-specific utilities if needed
from src.utils import load_pg_data  # noqa: E402


@click.command()
@click.option("--pgsql-file", type=click.Path(exists=True))
@click.option("--duck-path")
def main(pgsql_file, duck_path):
    query = Path(pgsql_file).read_text()
    df = load_pg_data(query)
    print(df.head())
    cond = duckdb.connect(duck_path)
    cond.register("raw", df)
    cond.execute("""CREATE OR REPLACE TABLE energy_features AS
      SELECT * FROM raw""")
    cond.unregister("raw")
    print(cond.execute("show tables;").fetchdf())
    print(cond.execute("SELECT * FROM energy_features LIMIT 5").fetchdf())


if __name__ == "__main__":
    main()
