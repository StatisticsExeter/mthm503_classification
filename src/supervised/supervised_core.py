import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)

from src.supervised.stubs import (
    stub_get_metrics_df, stub_generate_predictions, stub_compute_roc_data, stub_perform_split,
    stub_get_classifier, stub_train_model, stub_get_frequencies)
from src.utils import NotImplementedExercise, fallback_to


@fallback_to(stub_perform_split)
def perform_split(df, target_col):
    """
    Instructions:
    1. Drop any rows containing missing values from the input DataFrame.
    2. Separate the target column ('target_col') from the features.
    3. Perform a train-test split using a 70/30 split ratio.
    4. Ensure the split is stratified by the target variable.
    5. Use a random_state of 1999 for reproducibility.
    """
    raise NotImplementedExercise


@fallback_to(stub_get_classifier)
def get_classifier(model_type='lda'):
    """
    Instructions:
    1. Check if the 'model_type' string is 'lda' or 'qda'.
    2. If 'lda', return a LinearDiscriminantAnalysis object with priors set to [0.5, 0.5].
    3. If 'qda', return a QuadraticDiscriminantAnalysis object with priors set to [0.5, 0.5].
    Returns:
    An unfitted sklearn classifier object.
    """
    return NotImplementedExercise


@fallback_to(stub_train_model)
def train_model(X, y, model_obj):
    """
    Instructions:
    1. Take the provided feature DataFrame (X) and target Series (y).
    2. Fit the 'model_obj' using this data.
    3. Return the now-fitted model object.
    """
    return NotImplementedExercise


@fallback_to(stub_generate_predictions)
def generate_predictions(X, model):
    """
    Instructions:
    1. Use the fitted 'model' object to predict class labels for the feature matrix 'X'.
    2. Use the same 'model' to calculate the predicted probabilities for each class.
    3. Return a tuple containing:
       - The array of predicted class labels (y_pred).
       - The 2D array of predicted probabilities (y_prob).
    Parameters:
    X (pd.DataFrame): The feature matrix to predict on.
    model (sklearn.base.BaseEstimator): A fitted sklearn classifier.

    Returns:
    tuple: (y_pred, y_prob)
    """
    raise NotImplementedExercise


@fallback_to(stub_get_frequencies)
def get_frequencies(df, cat_column):
    raise NotImplementedExercise


@fallback_to(stub_get_metrics_df)
def get_metrics_df(y_test, y_pred):
    """
    Instructions:
    1. Use 'classification_report' from sklearn.metrics to generate a report.
    2. Crucial: Set 'output_dict=True' to ensure the result is a dictionary.
    3. Convert this dictionary into a pandas DataFrame.
    4. Transpose the DataFrame so that the metrics (precision, recall, etc.)
       are columns and the classes/averages are rows.
    5. Round all numeric values in the DataFrame to 2 decimal places.

    Parameters:
    y_test (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
    pd.DataFrame: A formatted metrics report.
    """
    raise NotImplementedExercise


@fallback_to(stub_compute_roc_data)
def compute_roc_data(y_test, y_prob):
    """
    Instructions:
    1. Initialize a LabelEncoder to convert 'y_test' into numeric labels (0 and 1).
    2. Use 'roc_curve' from sklearn.metrics to calculate the False Positive Rate (fpr)
       and True Positive Rate (tpr).
       Note: Use the second column of 'y_prob' (probabilities for the positive class).
    3. Calculate the Area Under the Curve (AUC) using the 'auc' function.
    4. Return a pandas DataFrame with three columns: 'fpr', 'tpr', and 'auc'.
       (The 'auc' value should be the same for all rows).

    Parameters:
    y_test (array-like): Actual target labels (can be strings like 'old'/'new').
    y_prob (array-like): Predicted probabilities for the positive class.

    Returns:
    pd.DataFrame: DataFrame containing fpr, tpr, and the scalar auc score.
    """
    raise NotImplementedExercise


def plot_roc(classifier_data):
    """
    Plots a matplotlib ROC curve.
    Data format: (fpr, tpr, auc_score)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(classifier_data['fpr'], classifier_data['tpr'],
            label=f'Classifier (AUC = {classifier_data["auc"].iloc[0]:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
