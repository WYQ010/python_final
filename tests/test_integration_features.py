# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from spaceship.features import build_features
from spaceship.preprocessing import preprocess
from tests.data_factory import make_dummy_raw_data

@pytest.fixture
def features_data(monkeypatch):
    # 1. 拦截 load_sample_submission，模拟一个特定顺序的 test_ids 模板
    mock_submission = pd.DataFrame({
        "PassengerId": ["0003_01", "0001_01", "0002_01"], 
        "Transported": [True, False, True]
    })
    monkeypatch.setattr("spaceship.features.load_sample_submission", lambda: mock_submission)

    # 2. 构造合规的 clean 数据
    raw_train = make_dummy_raw_data(n_rows=5)
    raw_test = make_dummy_raw_data(n_rows=3)
    
    # 强行修改 PassengerId 以匹配 mock_submission
    raw_train["PassengerId"] = ["0001_01", "0001_02", "0002_01", "0004_01", "0005_01"]
    raw_test["PassengerId"] = ["0001_01", "0002_01", "0003_01"]
    
    train_clean = preprocess(raw_train)
    test_clean = preprocess(raw_test)
    
    return train_clean, test_clean, mock_submission

def test_features_contract_1_shapes(features_data):
    train_clean, test_clean, _ = features_data
    fs = build_features(train_clean, test_clean)
    # 契约1：行数与输入一致
    assert fs.X_train.shape[0] == len(train_clean)
    assert fs.X_test.shape[0] == len(test_clean)

def test_features_contract_2_columns_identical(features_data):
    train_clean, test_clean, _ = features_data
    fs = build_features(train_clean, test_clean)
    # 契约2：列完全相同且同序
    assert list(fs.X_train.columns) == list(fs.X_test.columns)

def test_features_contract_3_no_nan_no_object(features_data):
    train_clean, test_clean, _ = features_data
    fs = build_features(train_clean, test_clean)
    # 契约3：无缺失值，且无 object 类型列（全为 int/float/bool/uint）
    assert fs.X_train.isna().sum().sum() == 0
    assert fs.X_test.isna().sum().sum() == 0
    assert len(fs.X_train.select_dtypes(include=['object']).columns) == 0
    assert len(fs.X_test.select_dtypes(include=['object']).columns) == 0

def test_features_contract_4_target_y(features_data):
    train_clean, test_clean, _ = features_data
    fs = build_features(train_clean, test_clean)
    # 契约4：y_train 必须为 bool，且与 clean 数据的 Transported 逐元素一致
    assert fs.y_train.dtype == bool
    pd.testing.assert_series_equal(
        fs.y_train.reset_index(drop=True), 
        train_clean["Transported"].reset_index(drop=True)
    )

def test_features_contract_5_test_ids_alignment(features_data):
    train_clean, test_clean, mock_submission = features_data
    fs = build_features(train_clean, test_clean)
    # 契约5：test_ids 与 sample_submission 完全一致且同序
    pd.testing.assert_series_equal(
        fs.test_ids.reset_index(drop=True), 
        mock_submission["PassengerId"].astype(str).reset_index(drop=True)
    )

def test_features_contract_6_determinism(features_data):
    train_clean, test_clean, _ = features_data
    # 契约6：确定性（同种子/同数据重跑结果一致）
    fs1 = build_features(train_clean, test_clean)
    fs2 = build_features(train_clean, test_clean)
    
    pd.testing.assert_frame_equal(fs1.X_train, fs2.X_train)
    pd.testing.assert_frame_equal(fs1.X_test, fs2.X_test)
    pd.testing.assert_series_equal(fs1.y_train, fs2.y_train)
    pd.testing.assert_series_equal(fs1.test_ids, fs2.test_ids)