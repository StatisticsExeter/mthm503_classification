import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def stub_perform_split(df, target_col):
    return (df.drop(columns=[target_col]), df.drop(columns=[target_col]),
            df[target_col], df[target_col])


def stub_classifier(model_type):
    return LinearDiscriminantAnalysis(priors=[0.5, 0.5])


def stub_train_model(X, y, model_obj):
    return model_obj.fit(X.iloc[:5], y.iloc[:5])


def stub_generate_predictions(X, model):
    n = len(X)

    y_pred = ['placeholder'] * n

    y_prob = pd.DataFrame({
        0: [0.5] * n,
        1: [0.5] * n
    }).to_numpy()
    return y_pred, y_prob


def stub_get_classifier(model_type):
    return LinearDiscriminantAnalysis(priors=[0.5, 0.5])


def stub_get_frequencies(df, cat_column):
    return pd.Series(1, index=df[cat_column].unique())


def stub_get_metrics_df(y_test, y_pred):
    return pd.DataFrame({
        "precision": [0.50],
        "recall": [0.50],
        "f1-score": [0.50],
        "support": [len(y_test)]},
        index=["placeholder"])


def stub_compute_roc_data(y_test, y_prob):
    return pd.DataFrame({
        "fpr": [0, 1],
        "tpr": [0, 1],
        "auc": [0.5, 0.5]})
