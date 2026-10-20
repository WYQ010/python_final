# -*- coding: utf-8 -*-
"""端到端流程编排（《统一接口定义》§3.6，负责人 B）。"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from spaceship.config import get_config
from spaceship.data_loader import load_test, load_train
from spaceship.features import FeatureSet, build_features
from spaceship.predict import make_submission, predict, validate_submission
from spaceship.preprocessing import preprocess
from spaceship.train import save_model, train_models
from spaceship.utils import set_global_seed


@dataclass
class PipelineReport:
    """封装端到端流程的模型、验证分数、生成文件和耗时。"""

    model_name: str
    cv_accuracy: float
    cv_detail: pd.DataFrame
    submission_path: Path | None
    model_path: Path | None
    elapsed_seconds: float


def _configure_logging() -> None:
    """仅在根日志器尚未配置时初始化一次项目日志。"""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def _load_feature_set() -> FeatureSet:
    """通过 A 轴公共 API 读数、清洗并构建模型特征。"""
    logger = logging.getLogger(__name__)
    train_raw = load_train()
    test_raw = load_test()
    logger.info("读取数据完成：train=%s，test=%s。", train_raw.shape, test_raw.shape)

    train_clean = preprocess(train_raw)
    test_clean = preprocess(test_raw)
    logger.info("预处理完成：train=%s，test=%s。", train_clean.shape, test_clean.shape)

    feature_set = build_features(train_clean, test_clean)
    logger.info(
        "特征构建完成：X_train=%s，X_test=%s。",
        feature_set.X_train.shape,
        feature_set.X_test.shape,
    )
    return feature_set


def run_full_pipeline(
    *,
    model_name: str | None = None,
    save_model_file: bool = True,
    cv_only: bool = False,
) -> PipelineReport:
    """按统一入口执行读数、特征、训练、预测与提交校验。

    参数：model_name 为空时使用默认模型；save_model_file 控制是否持久化模型；cv_only 为真时只训练和 CV。
    返回：PipelineReport；提交文件终检失败时抛出 ValueError。
    """
    _configure_logging()
    logger = logging.getLogger(__name__)
    cfg = get_config()
    set_global_seed(cfg.seed)
    started_at = perf_counter()

    feature_set = _load_feature_set()
    result = train_models(
        feature_set.X_train,
        feature_set.y_train,
        model_name=model_name,
    )
    logger.info("模型 %s 的 5 折 CV Accuracy：%.6f。", result.model_name, result.cv_accuracy)

    model_path = save_model(result) if save_model_file else None
    if model_path is not None:
        logger.info("模型已保存：%s。", model_path)

    submission_path: Path | None = None
    if not cv_only:
        predictions = predict(result.estimator, feature_set.X_test)
        submission_path = make_submission(
            feature_set.test_ids,
            predictions,
            model_name=result.model_name,
        )
        issues = validate_submission(submission_path, expected_ids=feature_set.test_ids)
        if issues:
            raise ValueError("提交文件终检失败：" + "；".join(issues))
        logger.info("提交文件终检通过：%s。", submission_path)

    elapsed_seconds = perf_counter() - started_at
    return PipelineReport(
        model_name=result.model_name,
        cv_accuracy=result.cv_accuracy,
        cv_detail=result.cv_detail,
        submission_path=submission_path,
        model_path=model_path,
        elapsed_seconds=elapsed_seconds,
    )
