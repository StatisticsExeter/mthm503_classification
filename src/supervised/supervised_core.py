import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.supervised.stubs import (
    stub_get_metrics_df, stub_generate_predictions, stub_compute_roc_data, stub_perform_split,
    stub_get_classifier, stub_train_model, stub_prepare_features, stub_encode_labels)
from src.utils import NotImplementedExercise, fallback_to


"""
Available classification models.

Keys are model names used throughout the workflow and values are
unfitted scikit-learn classifier objects configured with sensible
defaults.

To add a new model:
1. Import the required classifier from scikit-learn.
2. Add a new key-value pair to this dictionary.
3. Use the key when calling get_classifier().

The random_state argument is used where available to ensure
reproducible results.
"""
MODELS = {
    "lda": LinearDiscriminantAnalysis(priors=[0.5, 0.5]),
    "qda": QuadraticDiscriminantAnalysis(priors=[0.5, 0.5]),
    "logistic": LogisticRegression(max_iter=1000, random_state=42),
    "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
}


@fallback_to(stub_prepare_features)
def prepare_features(df, target_var):
    """
    Instructions:
    1. Separate the target variable from the predictor variables.
    2. Store the target variable in y.
    3. Store all remaining columns in X_PRE.
    4. Convert any categorical predictor variables into dummy variables
       using pd.get_dummies().
    5. Return X and y.
    """
    raise NotImplementedExercise


@fallback_to(stub_perform_split)
def perform_split(X, y):
    """
    Instructions:
    1. Split the data into training and test sets using a 70/30 ratio.
    2. Ensure the split is stratified using the target variable y.
    3. Use a random_state of 1999 for reproducibility.
    4. Return X_train, X_test, y_train and y_test.
    """
    raise NotImplementedExercise


@fallback_to(stub_get_classifier)
def get_classifier(model_type):
    """
    Return an unfitted classifier selected from the MODELS dictionary.
    Parameters
    ----------
    model_type : str
        The name of the classifier to retrieve. Must be a key in
        the MODELS dictionary.
    Returns
    -------
    sklearn.base.ClassifierMixin
        The corresponding scikit-learn classifier object.
    Raises
    ------
    ValueError
        If model_type is not a recognised model name.
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


@fallback_to(stub_encode_labels)
def encode_labels(y):
    """
    Convert categorical class labels into numeric labels.
    Instructions:
    1. Create a LabelEncoder object from sklearn.preprocessing.
    2. Fit the encoder to the input labels 'y'.
    3. Transform the labels into numeric values.
    4. Return the encoded labels.
    Parameters:
    y (array-like): Target labels to encode. Labels may be strings
    such as 'old'/'new' or 'A'/'B'.
    Returns:
    np.ndarray: Encoded labels represented as integers (e.g. 0 and 1
    for a binary classification problem).
    """
    raise NotImplementedExercise




@fallback_to(stub_compute_roc_data)
def compute_roc_data(y_test, y_prob):
    """
    Instructions:
    1. Use 'roc_curve' from sklearn.metrics to calculate the False Positive Rate (fpr)
       and True Positive Rate (tpr).
       Note: Use the second column of 'y_prob' (probabilities for the positive class).
    2. Calculate the Area Under the Curve (AUC) using the 'auc' function.
    3. Return a pandas DataFrame with three columns: 'fpr', 'tpr', and 'auc'.
       (The 'auc' value should be the same for all rows).

    Parameters:
    y_test_encoded (array-like): Actual target labels for test data
                                 encoded as integers (e.g., 0 and 1).
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


def scatter_by_group(df, group_var, title):
    """
    Create scatterplots of all pairs of numeric variables.
    Points are coloured and marked according to group_var.
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    group_var : str
        Name of the grouping variable.
    title : str
        Figure title.
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing scatterplots. Returns a figure with
        an explanatory message if fewer than two continuous
        variables are available.
    """
    numeric_cols = (df.select_dtypes(include="number")
                    .drop(columns=[group_var], errors="ignore").columns.tolist())
    if len(numeric_cols) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough continuous variables for scatterplots",
                ha="center", va="center", fontsize=14,)
        fig.suptitle(title)
        return fig
    pairs = list(combinations(numeric_cols, 2))
    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False,)
    axes = axes.flatten()
    markers = ["o", "s", "^", "D", "v", "P"]
    colours = plt.cm.tab10.colors
    groups = df[group_var].dropna().unique()
    for ax, (x, y) in zip(axes, pairs):
        for i, group in enumerate(groups):
            group_df = df[df[group_var] == group]
            ax.scatter(group_df[x], group_df[y], label=str(group),
                       marker=markers[i % len(markers)],
                       color=colours[i % len(colours)], alpha=0.7,)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    for ax in axes[len(pairs):]:
        ax.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right",)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
