# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import pytest

from spaceship.pipeline import run_full_pipeline
from spaceship.predict import validate_submission
from spaceship.config import get_config


DATA_DIR = get_config().paths.data_dir
HAS_DATA = (DATA_DIR / "train.csv").exists() and (DATA_DIR / "test.csv").exists()

@pytest.mark.skipif(not HAS_DATA, reason="本地缺少真实 data/，跳过 E2E 测试")
def test_e2e_full_pipeline_with_real_data():
    
    report = run_full_pipeline(model_name="logreg", save_model_file=False, cv_only=False)
    
    assert report.model_name == "logreg"
    assert report.cv_accuracy > 0.5 
    assert report.submission_path is not None
    
    
    issues = validate_submission(report.submission_path)
    assert len(issues) == 0