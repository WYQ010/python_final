# -*- coding: utf-8 -*-
"""集中配置：路径 / 随机种子 / 交叉验证 / 填补策略 / 模型参数。
结构冻结（《统一接口定义》§5）：只允许修改各字段的值，不得增删字段或改变结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    submission_dir: Path = PROJECT_ROOT / "outputs" / "submissions"
    model_dir: Path = PROJECT_ROOT / "outputs" / "models"
    report_dir: Path = PROJECT_ROOT / "outputs" / "reports"


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True


@dataclass(frozen=True)
class PreprocessConfig:
    age_fill: str = "median"             # median | mode | zero
    spend_fill: str = "cryo_then_zero"   # CryoSleep=True→0；其余→0（众数）
    homeplanet_fill: str = "mode"        # mode | unknown
    destination_fill: str = "mode"
    vip_fill: bool = False
    cryosleep_rule: str = "spend_zero"   # 缺失且 5 项消费全 0 → True，否则 False
    cabin_placeholder: str = "U/-1/U"
    name_placeholder: str = ""


@dataclass(frozen=True)
class ModelConfig:
    default_model: str = "logreg"
    params: dict = field(default_factory=lambda: {
        "logreg": {"max_iter": 1000, "solver": "newton-cholesky"},
        "hist_gb": None,                                   # None = 全默认
        "xgboost": {"n_estimators": 300, "learning_rate": 0.05},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05},
    })


@dataclass(frozen=True)
class Config:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    models: ModelConfig = field(default_factory=ModelConfig)


CONFIG = Config()


def get_config() -> Config:
    """返回全局唯一配置实例；全模块经此读取，禁止另行构造 Config。"""
    return CONFIG
