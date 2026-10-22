import click
import joblib
import pandas as pd
from pathlib import Path
import sys
# The next few lines ensure that the project root is in the Python path,
# allowing us to import modules from the project.
script_path = Path(__file__).resolve()
PROJECT_ROOT = script_path.parents[2]
if (str(PROJECT_ROOT) not in sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
from src.supervised.supervised_core import (  # noqa: E402
    get_classifier, train_model, generate_predictions, MODELS)
from src.utils import FALLBACKS_USED, report_fallbacks_used



@click.group()
def cli():
    pass


@cli.command()
@click.option('--x-train', type=click.Path(exists=True))
@click.option('--y-train', type=click.Path(exists=True))
@click.option('--model-type', type=click.Choice(sorted(MODELS.keys())))
@click.option('--model-out', type=click.Path())
def train(x_train, y_train, model_type, model_out):
    clf = get_classifier(model_type)
    X = pd.read_csv(x_train)
    y = pd.read_csv(y_train).iloc[:, 0]
    trained_model = train_model(X, y, clf)
    joblib.dump(trained_model, model_out)
    report_fallbacks_used()

@cli.command()
@click.option('--x-test', type=click.Path(exists=True))
@click.option('--model-in', type=click.Path(exists=True))
@click.option('--pred-out', type=click.Path())
@click.option('--prob-out', type=click.Path())
def predict(x_test, model_in, pred_out, prob_out):
    X = pd.read_csv(x_test)
    model = joblib.load(model_in)
    y_pred, y_prob = generate_predictions(X, model)
    pd.DataFrame(y_pred).to_csv(pred_out, index=False)
    pd.DataFrame(y_prob).to_csv(prob_out, index=False)
    report_fallbacks_used()


if __name__ == '__main__':
    cli()
