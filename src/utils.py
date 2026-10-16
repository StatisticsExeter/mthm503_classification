from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
import sys
import textwrap
import duckdb
from functools import wraps
import warnings
from click import secho as click_echo
import os
import pandas as pd
import psycopg2


FALLBACKS_USED = set()
load_dotenv()

class NotImplementedExercise(Exception):
    pass


def load_pg_data(query):
    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )

    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()



def load_duck_data(query, duck_file):
    ROOT = find_project_root()
    DB_PATH = ROOT / duck_file
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    with duckdb.connect(DB_PATH) as conn:
        return conn.execute(query).fetchdf()


def warn_if_in_rstudio():
    """Detect if running under RStudio/reticulate and print a warning."""
    # reticulate embeds Python in-process and sets these environment variables
    in_rstudio = any(
        "RSTUDIO" in k or "RETICULATE" in k for k in os.environ.keys()
    )

    if in_rstudio:
        msg = textwrap.dedent(f"""
        WARNING: This script appears to be running inside RStudio (reticulate).
        RStudio embeds Python in the R process, which can cause segfaults with
        libraries like psycopg2 and NumPy.

        Please re-run this script from a clean Anaconda prompt instead:
          conda activate python-exercises
          python {os.path.basename(sys.argv[0])}
        """)
        print(msg, file=sys.stderr)


def find_project_root(marker=".git"):
    """
    Traverse upward from the current file to find the project root.
    Includes the current file's directory in the search.
    """
    current_dir = Path.cwd().resolve()
    for directory in [current_dir] + list(current_dir.parents):
        if (directory / marker).exists():
            return directory
    raise RuntimeError("Project root not found. Please ensure a marker file exists.")


def prepare_collision_data(query_text):
    """
    Instructions:
    1. Load raw data from PostgreSQL using the provided query.
    2. Set the 'local_authority_ons_district' column as the DataFrame index.
    3. Convert the denominator (column 0) and numerator columns (columns 1-6) 
       to float to ensure precise division.
    4. Normalize the numerator columns by dividing them by the denominator column 
       (the 'total' or 'population' value).
    5. Return the processed DataFrame containing the normalized rates.

    Parameters:
    query_text (str): SQL query to fetch raw collision and population data.

    Returns:
    pd.DataFrame: A DataFrame where columns 1-6 represent normalized rates.
    """
    df = load_pg_data(query_text)
    df.set_index("local_authority_ons_district", inplace=True)
    # Convert and normalize
    df[df.columns[1:7]] = df[df.columns[1:7]].astype(float)
    df[df.columns[0]] = df[df.columns[0]].astype(float)
    df.iloc[:, 1:7] = df.iloc[:, 1:7].div(df.iloc[:, 0], axis=0)
    df_clustering = df.iloc[:, 1:]
    return df, df_clustering


def fallback_to(fake_func):
    def decorator(real_func):
        @wraps(real_func)
        def wrapper(*args, **kwargs):
            try:
                return real_func(*args, **kwargs)
            except NotImplementedExercise:
                warnings.warn(
                    f"{real_func.__name__} has not been implemented; using fallback implementation instead.",
                    UserWarning, stacklevel=2)
                FALLBACKS_USED.add(f"{real_func.__module__}.{real_func.__name__}")
                return fake_func(*args, **kwargs)
        return wrapper
    return decorator

def report_fallbacks_used():
    if not FALLBACKS_USED:
        return
    click_echo("\n=== FALLBACK IMPLEMENTATIONS USED ===============================",
                fg="red", bold=True,)
    print("\n=== THESE FUNCTIONS HAVE TO BE IMPLEMENTED FOR THE COURSEWORK ===")
    for name in sorted(FALLBACKS_USED):
        print(f" - {name}")
