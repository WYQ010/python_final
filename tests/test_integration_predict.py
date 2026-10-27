# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from spaceship.predict import predict, make_submission, validate_submission

class DummyModel:
    def predict(self, X):
        return [True, False] * (len(X) // 2) + [True] * (len(X) % 2)

@pytest.fixture
def mock_predictions_setup(monkeypatch, tmp_path):
    class MockPaths:
        submission_dir = tmp_path
    
    class MockConfig:
        paths = MockPaths()
        
    monkeypatch.setattr("spaceship.predict.get_config", lambda: MockConfig())
    
    test_ids = pd.Series([f"{i:04d}_01" for i in range(1, 4278)], dtype=str)
    predictions = pd.Series([True, False] * 2138 + [True], dtype=bool)
    
    return test_ids, predictions

def test_predict_returns_aligned_bool_series():
    model = DummyModel()
    X_test = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
    
    result = predict(model, X_test)
    assert result.dtype == bool
    assert result.index.equals(X_test.index)

def test_make_submission_valid(mock_predictions_setup):
    test_ids, predictions = mock_predictions_setup
    path = make_submission(test_ids, predictions, model_name="test_model")
    
    assert path.exists()
    assert path.name.startswith("submission_test_model_")
    
    df = pd.read_csv(path)
    assert list(df.columns) == ["PassengerId", "Transported"]
    assert len(df) == 4277

def test_make_submission_invalid_length(mock_predictions_setup):
    test_ids, predictions = mock_predictions_setup
    bad_predictions = predictions.iloc[:-1]
    
    with pytest.raises(ValueError, match="预测行数应为 4277"):
        make_submission(test_ids, bad_predictions)

def test_make_submission_invalid_alignment(mock_predictions_setup):
    test_ids, predictions = mock_predictions_setup
    bad_predictions = predictions.copy()
    bad_predictions.index = range(1, 4278)
    
    with pytest.raises(ValueError, match="PassengerId 与预测值的索引不一致"):
        make_submission(test_ids, bad_predictions)

def test_validate_submission_valid(mock_predictions_setup):
    test_ids, predictions = mock_predictions_setup
    path = make_submission(test_ids, predictions)
    
    issues = validate_submission(path, expected_ids=test_ids)
    assert issues == []

def test_validate_submission_missing_file():
    issues = validate_submission("non_existent_file.csv")
    assert len(issues) == 1
    assert "提交文件不存在" in issues[0]

def test_validate_submission_bad_format(mock_predictions_setup, tmp_path):
    test_ids, _ = mock_predictions_setup
    bad_file = tmp_path / "bad.csv"
    
    bad_df = pd.DataFrame({"ID": test_ids.iloc[:10], "Target": ["Yes"] * 10})
    bad_df.to_csv(bad_file, index=False)
    
    issues = validate_submission(bad_file)
    assert len(issues) > 0
    assert any("列名及顺序必须为" in i for i in issues)