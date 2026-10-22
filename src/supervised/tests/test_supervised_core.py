import pandas as pd
import numpy as np
import pytest  # noqa: F401
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from src.supervised.supervised_core import (
    perform_split, get_classifier, train_model, generate_predictions,
    get_metrics_df, compute_roc_data, prepare_features, encode_labels)


def test_perform_split_stratification():
    df = pd.DataFrame({'feature': np.random.rand(100),
                       'built_age': [0] * 70 + [1] * 30})
    y = df['built_age']
    X = df.drop(columns=['built_age'])
    X_train, X_test, y_train, y_test = perform_split(X, y)
    assert len(X_test) == 30
    # Original proportion = 30%
    assert abs(y_train.mean() - 0.30) < 0.05
    assert abs(y_test.mean() - 0.30) < 0.05
    # Stratification should preserve both classes
    assert len(y_train.unique()) == 2


def test_prepare_features_creates_dummies():
    df = pd.DataFrame({
        "colour": ["red", "blue", "red"],
        "target": [0, 1, 0]
    })
    X, y = prepare_features(df, "target")
    assert "colour_blue" in X.columns
    assert "colour_red" in X.columns
    assert list(y) == [0, 1, 0]


def test_get_classifier_types():
    # Test LDA selection
    lda_model = get_classifier('lda')
    assert isinstance(lda_model, LinearDiscriminantAnalysis)
    assert lda_model.priors == [0.5, 0.5]
    # Test QDA selection
    qda_model = get_classifier('qda')
    assert isinstance(qda_model, QuadraticDiscriminantAnalysis)
    assert qda_model.priors == [0.5, 0.5]
    logistic_model = get_classifier('logistic')
    assert isinstance(logistic_model, type(get_classifier('logistic')))
    assert logistic_model.get_params()['max_iter'] == 1000
    tree_model = get_classifier('tree')
    assert isinstance(tree_model, type(get_classifier('tree')))
    assert tree_model.get_params()['max_depth'] == 5


def test_get_classifier_raises_value_error_for_unknown_model():
    with pytest.raises(ValueError) as exc_info:
        get_classifier("not_a_real_model")

    assert str(exc_info.value) == "Unknown model_type: not_a_real_model"


def test_train_model_logic():
    # Create simple dummy data
    X = pd.DataFrame({'feat': [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    clf = LinearDiscriminantAnalysis()
    # Fit model using our function
    fitted_clf = train_model(X, y, clf)
    # Check if the model is actually fitted
    # (Scikit-learn models add an underscore to attributes after fitting)
    assert hasattr(fitted_clf, 'classes_'), "The model was not fitted."
    assert list(fitted_clf.classes_) == [0, 1]


def test_generate_predictions_shapes():
    # 1. Setup a fitted model and dummy data
    X_train = pd.DataFrame({'feat': [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 1, 1])
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    # New data to predict
    X_new = pd.DataFrame({'feat': [1.5, 3.5]})
    # 2. Run the function
    y_pred, y_prob = generate_predictions(X_new, model)
    # 3. Assertions
    # Check that y_pred has the same number of rows as X_new
    assert len(y_pred) == 2
    # Check that y_prob is 2D and has 2 columns (one for class 0, one for class 1)
    assert y_prob.ndim == 2
    assert y_prob.shape[1] == 2
    # Check that probabilities sum to approximately 1
    assert np.allclose(y_prob.sum(axis=1), 1.0)


def test_generate_predictions_values():
    # Simple check to ensure output values are within expected ranges
    X = pd.DataFrame({'feat': [1, 2, 3, 4, 5]})
    y = pd.Series([0, 0, 1, 1, 1])
    model = LinearDiscriminantAnalysis().fit(X, y)
    y_pred, y_prob = generate_predictions(X, model)
    # Predictions should match training data for this simple case
    assert list(y_pred) == [0, 0, 1, 1, 1]
    # Probabilities must be between 0 and 1
    assert np.all((y_prob >= 0) & (y_prob <= 1))


def test_get_metrics_df_structure():
    # Setup simple binary labels
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]  # One mistake (False Positive)
    df_metrics = get_metrics_df(y_true, y_pred)
    # 1. Check if the output is a DataFrame
    assert isinstance(df_metrics, pd.DataFrame)
    # 2. Check for the expected rows in the transposed report
    # Scikit-learn includes class labels ('0', '1') and averages
    expected_rows = ['0', '1', 'accuracy', 'macro avg', 'weighted avg']
    for row in expected_rows:
        assert row in df_metrics.index
    # 3. Check for the expected metric columns
    expected_cols = ['precision', 'recall', 'f1-score', 'support']
    for col in expected_cols:
        assert col in df_metrics.columns


def test_get_metrics_df_rounding():
    # Test that rounding to 2 decimal places is actually happening
    y_true = [0, 0, 1]
    y_pred = [0, 1, 1]
    df_metrics = get_metrics_df(y_true, y_pred)
    # Check a value that would typically have many decimals
    # Precision for class 1 here is 0.5; Precision for class 0 is 1.0
    # Let's check the 'macro avg' precision: (1.0 + 0.5) / 2 = 0.75
    # If the student didn't round, it might be 0.7500000000001
    val = df_metrics.loc['macro avg', 'precision']
    assert str(val) == str(round(val, 2))


def test_compute_roc_data_logic():
    # 1. Setup string labels and mock probabilities
    # Let's assume 'built_age' classes are 'historic' and 'modern'
    y_test = [0, 1, 0, 1]
    # Perfect predictions: historic=0.1, modern=0.9
    y_prob = [0.1, 0.9, 0.2, 0.8]
    result = compute_roc_data(y_test, y_prob)
    # 2. Check return type and columns
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['fpr', 'tpr', 'auc']
    # 3. Verify AUC is a valid probability (0 to 1)
    # With perfect separation like above, AUC should be 1.0
    assert result['auc'].iloc[0] == 1.0
    # 4. Verify FPR and TPR boundaries
    # An ROC curve must always start at (0,0) and end at (1,1)
    assert result['fpr'].iloc[0] == 0.0
    assert result['tpr'].iloc[-1] == 1.0


def test_encode_labels():
    y = ["B", "A", "B"]
    assert list(encode_labels(y)) == [1, 0, 1]
