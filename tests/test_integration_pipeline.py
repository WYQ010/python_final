# -*- coding: utf-8 -*-
import logging
import pandas as pd
import pytest
from pathlib import Path

from spaceship.pipeline import run_full_pipeline, PipelineReport
from tests.data_factory import make_dummy_raw_data

@pytest.fixture
def fake_pipeline_env(monkeypatch, tmp_path):
    # 【核心修复1】必须生成 4277 行的测试数据，否则过不了 predict 里的行数校验
    raw_train = make_dummy_raw_data(n_rows=100)  # 训练集可以小一点，跑得快
    raw_test = make_dummy_raw_data(n_rows=4277)  # 测试集必须严格等于 4277 行
    
    raw_test["PassengerId"] = [f"{i:04d}_01" for i in range(10000, 10000 + 4277)]
    raw_test = raw_test.drop(columns=["Transported"])
    
    fake_sub = pd.DataFrame({
        "PassengerId": raw_test["PassengerId"], 
        "Transported": [True] * 4277
    })
    
    # 路径计算
    temp_data_dir = tmp_path / "data"
    temp_output_dir = tmp_path / "outputs"
    temp_submission_dir = temp_output_dir / "submissions"
    temp_model_dir = temp_output_dir / "models"
    temp_report_dir = temp_output_dir / "reports"

    temp_data_dir.mkdir()
    raw_train.to_csv(temp_data_dir / "train.csv", index=False)
    raw_test.to_csv(temp_data_dir / "test.csv", index=False)
    fake_sub.to_csv(temp_data_dir / "sample_submission.csv", index=False)

    # 拦截配置类的路径
    from spaceship.config import get_config
    real_cfg = get_config()
    
    class MockPaths:
        data_dir = temp_data_dir
        output_dir = temp_output_dir
        submission_dir = temp_submission_dir
        model_dir = temp_model_dir
        report_dir = temp_report_dir

    class MockConfig:
        seed = 42
        cv = real_cfg.cv
        preprocess = real_cfg.preprocess
        models = real_cfg.models
        paths = MockPaths()

    monkeypatch.setattr("spaceship.pipeline.get_config", lambda: MockConfig())
    monkeypatch.setattr("spaceship.data_loader.get_config", lambda: MockConfig())
    monkeypatch.setattr("spaceship.predict.get_config", lambda: MockConfig())
    monkeypatch.setattr("spaceship.train.get_config", lambda: MockConfig())

def test_pipeline_run_full_pipeline(fake_pipeline_env, caplog):
    caplog.set_level(logging.INFO)
    
    report = run_full_pipeline(model_name="logreg", save_model_file=True, cv_only=False)
    
    assert isinstance(report, PipelineReport)
    assert report.model_name == "logreg"
    assert isinstance(report.cv_accuracy, float)
    assert isinstance(report.cv_detail, pd.DataFrame)
    assert report.elapsed_seconds > 0
    
    assert report.model_path is not None and report.model_path.exists()
    assert report.submission_path is not None and report.submission_path.exists()
    
    sub_df = pd.read_csv(report.submission_path)
    assert list(sub_df.columns) == ["PassengerId", "Transported"]
    assert len(sub_df) == 4277
    
    # 验证契约要求的日志输出
    assert "读取数据完成" in caplog.text
    assert "预处理完成" in caplog.text
    assert "特征构建完成" in caplog.text
    assert "提交文件终检通过" in caplog.text

def test_pipeline_cv_only(fake_pipeline_env):
    # 【核心修复2】显式传入 save_model_file=False，才能真正不保存模型
    report = run_full_pipeline(model_name="logreg", save_model_file=False, cv_only=True)
    
    assert report.model_path is None
    assert report.submission_path is None