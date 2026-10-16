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
@click.option("--duck-path", type=click.Path(exists=True))
def main(pgsql_file, duck_path):
    query = Path(pgsql_file).read_text()
    df = load_pg_data(query)
    print(df.head())
    cond = duckdb.connect(duck_path)
    cond.register("raw", df)
    cond.execute("""CREATE OR REPLACE TABLE collision_features AS
      SELECT
      local_authority_ons_district, n_miniroundabouts, n_raining, n_dark, n_dry, n_urban, n_slight,
      n_miniroundabouts * 1.0 / n_total AS p_miniroundabouts,
      n_raining * 1.0 / n_total AS p_raining,
      n_dark * 1.0 / n_total AS p_dark,
      n_dry * 1.0 / n_total AS p_dry,
      n_urban * 1.0 / n_total AS p_urban,
      n_slight * 1.0 / n_total AS p_slight
  FROM raw""")
    print(cond.execute("show tables;").fetchdf())
    print(cond.execute("SELECT * FROM collision_features LIMIT 5").fetchdf())


if __name__ == "__main__":
    main()
