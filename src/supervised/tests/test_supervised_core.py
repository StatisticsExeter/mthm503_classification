import pandas as pd
import numpy as np
import pytest  # noqa: F401
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from src.supervised.supervised_core import (
    perform_split, get_classifier, train_model, generate_predictions,
    get_metrics_df, compute_roc_data)


def test_perform_split_stratification():
    # Create an imbalanced dataset
    df = pd.DataFrame({
        'feature': np.random.rand(10),
        'built_age': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    })
    X_train, X_test, y_train, y_test = perform_split(df, 'built_age')
    # Test 30% split (10 rows * 0.3 = 3 rows in test)
    assert len(X_test) == 3
    # Check stratification: y_train should have both classes
    assert len(y_train.unique()) == 2


def test_get_classifier_types():
    # Test LDA selection
    lda_model = get_classifier('lda')
    assert isinstance(lda_model, LinearDiscriminantAnalysis)
    assert lda_model.priors == [0.5, 0.5]
    # Test QDA selection
    qda_model = get_classifier('qda')
    assert isinstance(qda_model, QuadraticDiscriminantAnalysis)
    assert qda_model.priors == [0.5, 0.5]


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
    y_test = ['historic', 'modern', 'historic', 'modern']
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


def test_compute_roc_data_encoding():
    # Ensure the LabelEncoder is handling the transformation correctly
    y_test = ['B', 'A', 'B']
    y_prob = [0.6, 0.2, 0.7]
    # If the function fails to encode 'A' and 'B', roc_curve will throw a ValueError
    try:
        _ = compute_roc_data(y_test, y_prob)
    except ValueError as e:
        pytest.fail(f"compute_roc_data failed to encode string labels: {e}")
