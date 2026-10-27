# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from spaceship.train import train_models, MODEL_REGISTRY, TrainResult

@pytest.fixture
def dummy_train_data():
    
    X_train = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    y_train = pd.Series([True, False, True, False, True, False, True, False, True, False], name="Transported")
    return X_train, y_train

def test_train_registry_has_required_models():

    assert "logreg" in MODEL_REGISTRY
    gbdt_models = ["hist_gb", "xgboost", "lightgbm"]
    assert any(m in MODEL_REGISTRY for m in gbdt_models)

def test_train_unknown_model_raises_error(dummy_train_data):
    X_train, y_train = dummy_train_data

    with pytest.raises(KeyError, match="可用模型："):
        train_models(X_train, y_train, model_name="non_existent_model")

def test_train_result_structure(dummy_train_data):
    X_train, y_train = dummy_train_data
    result = train_models(X_train, y_train, model_name="logreg")
    

    assert isinstance(result, TrainResult)
    assert result.model_name == "logreg"
    assert hasattr(result.estimator, "predict")
    assert isinstance(result.cv_accuracy, float)
    assert isinstance(result.cv_detail, pd.DataFrame)
    assert result.seed == 42
    assert isinstance(result.fitted_at, str)

def test_train_reproducibility(dummy_train_data):
    X_train, y_train = dummy_train_data
    
    result1 = train_models(X_train, y_train, model_name="logreg")
    result2 = train_models(X_train, y_train, model_name="logreg")
    
    assert result1.cv_accuracy == result2.cv_accuracy
    pd.testing.assert_frame_equal(result1.cv_detail, result2.cv_detail)