# -*- coding: utf-8 -*-
"""模型训练、交叉验证与模型持久化（《统一接口定义》§3.4，负责人 B）。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from spaceship.config import get_config
from spaceship.utils import set_global_seed


@dataclass
class TrainResult:
    """封装已训练模型及其交叉验证结果。"""

    model_name: str
    estimator: Any
    cv_accuracy: float
    cv_detail: pd.DataFrame
    seed: int
    fitted_at: str


def _model_params(model_name: str) -> dict[str, Any]:
    """读取指定模型的配置参数并返回副本。"""
    params = get_config().models.params.get(model_name)
    return dict(params or {})


def _build_logreg(model_name: str = "logreg") -> LogisticRegression:
    """构造固定随机种子的逻辑回归基线模型。"""
    cfg = get_config()
    params = _model_params(model_name)
    params.setdefault("random_state", cfg.seed)
    return LogisticRegression(**params)


def _build_hist_gb(model_name: str = "hist_gb") -> HistGradientBoostingClassifier:
    """构造固定随机种子的 sklearn 直方图梯度提升模型。"""
    cfg = get_config()
    params = _model_params(model_name)
    params.setdefault("random_state", cfg.seed)
    return HistGradientBoostingClassifier(**params)


def _build_xgboost(model_name: str = "xgboost") -> Any:
    """按需导入并构造 XGBoost 分类器，避免未使用时产生额外依赖。"""
    from xgboost import XGBClassifier

    cfg = get_config()
    params = _model_params(model_name)
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", 1)
    params.setdefault("eval_metric", "logloss")
    return XGBClassifier(**params)


def _build_lightgbm(model_name: str = "lightgbm") -> Any:
    """按需导入并构造 LightGBM 分类器，避免未使用时产生额外依赖。"""
    from lightgbm import LGBMClassifier

    cfg = get_config()
    params = _model_params(model_name)
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", 1)
    params.setdefault("verbosity", -1)
    return LGBMClassifier(**params)


def _build_rf() -> RandomForestClassifier:
    """构造固定随机种子的随机森林模型（集成成员，单线程保确定性）。"""
    cfg = get_config()
    params = _model_params("rf")
    params.setdefault("n_estimators", 500)
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", 1)
    return RandomForestClassifier(**params)


def _build_ensemble_soft() -> VotingClassifier:
    """按 config.models.params['ensemble_soft'] 组装软投票集成体。

    成员与权重全部来自配置（weights 为 None 时等权）；成员经注册表复用各自配置，
    随机源由各成员工厂显式 seed。
    """
    params = _model_params("ensemble_soft")
    members = params.get("members") or []
    if not members:
        raise ValueError(
            "ensemble_soft 配置缺少 members 列表；请在 config.models.params 中补全。"
        )
    estimators = [(name, _build_model(name)) for name in members]
    weights = params.get("weights")
    return VotingClassifier(
        estimators=estimators, voting="soft", weights=weights, n_jobs=1,
    )


def _build_stacking() -> StackingClassifier:
    """按 config.models.params['stacking'] 组装 OOF 堆叠集成体。

    一层成员的元特征由 StackingClassifier 内部交叉验证生成（cross_val_predict 式，
    不用全量拟合概率喂二层，无泄漏）；二层为逻辑回归。
    """
    cfg = get_config()
    params = _model_params("stacking")
    members = params.get("members") or []
    if not members:
        raise ValueError(
            "stacking 配置缺少 members 列表；请在 config.models.params 中补全。"
        )
    final_params = dict(params.get("final") or {"max_iter": 1000})
    estimators = [(name, _build_model(name)) for name in members]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(**final_params),
        cv=StratifiedKFold(
            n_splits=cfg.cv.n_splits, shuffle=cfg.cv.shuffle, random_state=cfg.seed,
        ),
        stack_method="predict_proba",
        n_jobs=1,
    )


class _KFoldProbaAverage(BaseEstimator, ClassifierMixin):
    """持有 K 个折模型、预测期取概率均值的包装器（投榜变体专用）。

    fit 在训练数据内部做 K 折切分并逐折克隆训练 base_estimator；
    predict_proba 返回各折模型概率的等权均值——与单模型相比 CV 口径
    基本不变，价值在测试期预测方差缩减（第六轮实验定位，非 CV 增益主张）。
    显式 seed、单线程，保证同种子逐折一致。
    """

    def __init__(self, base_estimator: Any, n_splits: int = 5, random_state: int = 42):
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_KFoldProbaAverage":
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state,
        )
        self.fold_models_: list[Any] = []
        for train_idx, _ in splitter.split(X, y):
            fold_model = clone(self.base_estimator)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            self.fold_models_.append(fold_model)
        self.classes_ = self.fold_models_[0].classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = [model.predict_proba(X) for model in self.fold_models_]
        return np.mean(probas, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _build_hist_gb_cg_avg() -> _KFoldProbaAverage:
    """构造 hist_gb（CG 参数）五折概率平均变体（参数见 config）。"""
    cfg = get_config()
    params = _model_params("hist_gb_cg_avg")
    return _KFoldProbaAverage(
        base_estimator=_build_hist_gb("hist_gb"),
        n_splits=int(params.get("n_splits", 5)),
        random_state=cfg.seed,
    )


MODEL_REGISTRY: dict[str, Callable[[], Any]] = {
    "logreg": _build_logreg,
    "hist_gb": _build_hist_gb,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
    "rf": _build_rf,
    # 随机化 GBDT 变体：同族不同种子/子采样，作为集成多样性成员（参数见 config）
    "lightgbm_sto_s42": partial(_build_lightgbm, "lightgbm_sto_s42"),
    "lightgbm_sto_s7": partial(_build_lightgbm, "lightgbm_sto_s7"),
    "lightgbm_sto_s123": partial(_build_lightgbm, "lightgbm_sto_s123"),
    "hist_gb_sto_s42": partial(_build_hist_gb, "hist_gb_sto_s42"),
    "hist_gb_sto_s7": partial(_build_hist_gb, "hist_gb_sto_s7"),
    "ensemble_soft": _build_ensemble_soft,
    "stacking": _build_stacking,
    # 五折概率平均投榜变体：CV 与 hist_gb 持平，测试期概率取五折模型均值
    "hist_gb_cg_avg": _build_hist_gb_cg_avg,
}


def _build_model(model_name: str) -> Any:
    """从注册表构造模型；不存在时给出全部可用名称。"""
    try:
        factory = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"未知模型名 {model_name!r}；可用模型：{available}。"
        ) from exc
    return factory()


def _make_cv_detail(scores: list[float]) -> pd.DataFrame:
    """把每折 Accuracy 与汇总统计整理为稳定的表格。"""
    rows: list[dict[str, int | str | float]] = [
        {"fold": fold, "accuracy": score}
        for fold, score in enumerate(scores, start=1)
    ]
    score_series = pd.Series(scores, dtype="float64")
    rows.extend(
        [
            {"fold": "mean", "accuracy": float(score_series.mean())},
            {"fold": "std", "accuracy": float(score_series.std(ddof=0))},
        ]
    )
    return pd.DataFrame(rows, columns=["fold", "accuracy"])


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_name: str | None = None,
) -> TrainResult:
    """训练一个注册模型并进行固定种子的分层五折交叉验证。

    参数：X_train 为全数值训练特征，y_train 为 bool 标签，model_name 为空时使用默认模型。
    返回：已拟合的 TrainResult；未知模型名时抛出含可用模型列表的 KeyError。
    """
    cfg = get_config()
    selected_name = model_name or cfg.models.default_model
    set_global_seed(cfg.seed)

    estimator = _build_model(selected_name)
    cv = StratifiedKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.seed,
    )
    scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    estimator.fit(X_train, y_train)

    score_values = [float(score) for score in scores]
    return TrainResult(
        model_name=selected_name,
        estimator=estimator,
        cv_accuracy=float(pd.Series(score_values, dtype="float64").mean()),
        cv_detail=_make_cv_detail(score_values),
        seed=cfg.seed,
        fitted_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def save_model(result: TrainResult, path: str | Path | None = None) -> Path:
    """把 TrainResult 序列化为 joblib 文件并返回写入路径。

    参数：result 为训练结果；path 为空时写入 config 的模型输出目录。
    返回：实际写入的模型文件路径。
    """
    cfg = get_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    target = (
        Path(path)
        if path is not None
        else cfg.paths.model_dir / f"model_{result.model_name}_{timestamp}.joblib"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, target)
    return target


def load_model(path: str | Path) -> TrainResult:
    """从 joblib 文件加载 TrainResult。

    参数：path 为模型文件路径。
    返回：反序列化的 TrainResult；文件不存在或内容类型错误时抛出含路径的异常。
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"未找到模型文件：{source}")
    result = joblib.load(source)
    if not isinstance(result, TrainResult):
        raise TypeError(f"模型文件 {source} 未保存 TrainResult，实际类型为 {type(result)!r}。")
    return result
