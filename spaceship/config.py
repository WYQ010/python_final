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
        # 2026-10-25 采纳：小叶+强正则组合（源自 ChatGPT 对照工程交叉验证，
        # 50 折配对 +0.0032 达采用线；全默认 0.8123 → 本组合 0.8144）
        "hist_gb": {"learning_rate": 0.06, "max_iter": 300, "max_leaf_nodes": 15,
                    "min_samples_leaf": 25, "l2_regularization": 3.0,
                    "early_stopping": False, "random_state": 42},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.05},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05},
        "rf": {"n_estimators": 500},                       # 集成成员：随机森林
        # 随机化 GBDT 变体（行/列子采样 × 不同种子）：同族低相关集成成员
        "lightgbm_sto_s42": {"n_estimators": 300, "learning_rate": 0.05,
                             "bagging_fraction": 0.8, "bagging_freq": 1,
                             "feature_fraction": 0.8, "random_state": 42},
        "lightgbm_sto_s7": {"n_estimators": 300, "learning_rate": 0.05,
                            "bagging_fraction": 0.8, "bagging_freq": 1,
                            "feature_fraction": 0.8, "random_state": 7},
        "lightgbm_sto_s123": {"n_estimators": 300, "learning_rate": 0.05,
                              "bagging_fraction": 0.8, "bagging_freq": 1,
                              "feature_fraction": 0.8, "random_state": 123},
        "hist_gb_sto_s42": {"max_features": 0.8, "random_state": 42},
        "hist_gb_sto_s7": {"max_features": 0.8, "random_state": 7},
        # 软投票集成：GBDT 系 7 成员 + rf + logreg 等权（10 折 OOF 筛选 + 多协议验证）
        "ensemble_soft": {
            "members": ["hist_gb", "lightgbm",
                        "lightgbm_sto_s42", "lightgbm_sto_s7", "lightgbm_sto_s123",
                        "hist_gb_sto_s42", "hist_gb_sto_s7",
                        "rf", "logreg"],
            "weights": None,                               # None = 等权
        },
        # OOF 堆叠：一层成员概率由内部 CV 生成（无泄漏），二层 logreg
        "stacking": {
            "members": ["hist_gb", "xgboost", "lightgbm", "logreg", "rf"],
            "final": {"max_iter": 1000},
        },
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
