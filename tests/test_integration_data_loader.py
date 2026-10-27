import pytest
from pathlib import Path
from spaceship.data_loader import load_train, load_test, load_sample_submission

def test_load_missing_file_raises_error(monkeypatch, tmp_path):
    class MockPaths:
        data_dir = tmp_path / "not_exist_data"
    class MockConfig:
        paths = MockPaths()
        
    monkeypatch.setattr("spaceship.data_loader.get_config", lambda: MockConfig())
    
    with pytest.raises(FileNotFoundError, match="未找到数据文件，期望路径："):
        load_train()
    with pytest.raises(FileNotFoundError, match="未找到数据文件，期望路径："):
        load_test()
    with pytest.raises(FileNotFoundError, match="未找到数据文件，期望路径："):
        load_sample_submission()

@pytest.mark.skipif(not Path("data/train.csv").exists(), reason="无真实数据")
def test_load_train_contract():
    df = load_train()
    assert df.shape == (8693, 14)
    assert len(df.columns) == 14
    assert not df["PassengerId"].duplicated().any()

@pytest.mark.skipif(not Path("data/test.csv").exists(), reason="无真实数据")
def test_load_test_contract():
    df = load_test()
    assert df.shape == (4277, 13)
    assert "Transported" not in df.columns
    assert not df["PassengerId"].duplicated().any()

@pytest.mark.skipif(not Path("data/sample_submission.csv").exists(), reason="无真实数据")
def test_load_sample_submission_contract():
    df = load_sample_submission()
    assert df.shape == (4277, 2)
    assert list(df.columns) == ["PassengerId", "Transported"]