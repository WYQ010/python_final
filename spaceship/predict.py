# -*- coding: utf-8 -*-
"""推理、提交文件写出与校验（《统一接口定义》§3.5，负责人 B）。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from spaceship.config import get_config

_SUBMISSION_COLUMNS = ["PassengerId", "Transported"]
_TEST_ROW_COUNT = 4277


def _as_bool_predictions(predictions: pd.Series) -> pd.Series:
    """校验预测值为二元标签并规范为 bool Series。"""
    if predictions.isna().any() or not predictions.isin([True, False, 0, 1]).all():
        raise ValueError("预测值必须全部属于 True/False（或等价的 0/1），且不得缺失。")
    return predictions.astype(bool).rename("Transported")


def predict(estimator: Any, X_test: pd.DataFrame) -> pd.Series:
    """使用已拟合估计器预测测试特征。

    参数：estimator 必须具备 predict(X) 接口，X_test 为特征矩阵。
    返回：与 X_test 索引对齐的 bool Series，不修改 X_test。
    """
    raw_predictions = pd.Series(estimator.predict(X_test), index=X_test.index)
    return _as_bool_predictions(raw_predictions)


def _submission_input_errors(test_ids: pd.Series, predictions: pd.Series) -> list[str]:
    """在落盘前校验提交所需的 ID 与预测序列。"""
    errors: list[str] = []
    if len(test_ids) != _TEST_ROW_COUNT:
        errors.append(f"PassengerId 行数应为 {_TEST_ROW_COUNT}，实际为 {len(test_ids)}。")
    if len(predictions) != _TEST_ROW_COUNT:
        errors.append(f"预测行数应为 {_TEST_ROW_COUNT}，实际为 {len(predictions)}。")
    if not test_ids.index.equals(predictions.index):
        errors.append("PassengerId 与预测值的索引不一致，无法保证逐行对齐。")
    if test_ids.isna().any():
        errors.append("PassengerId 不得缺失。")
    if test_ids.astype(str).duplicated().any():
        errors.append("PassengerId 不得重复。")
    try:
        _as_bool_predictions(predictions)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def make_submission(
    test_ids: pd.Series,
    predictions: pd.Series,
    *,
    model_name: str = "model",
) -> Path:
    """校验预测并写出 Kaggle 提交文件。

    参数：test_ids 为与模板同序的 PassengerId，predictions 为对齐的 bool 预测，model_name 用于命名。
    返回：写入 config 提交目录的 CSV 路径；输入违规时抛出 ValueError。
    """
    errors = _submission_input_errors(test_ids, predictions)
    if errors:
        raise ValueError("提交文件未生成：" + "；".join(errors))

    submission = pd.DataFrame(
        {
            "PassengerId": test_ids.astype(str),
            "Transported": _as_bool_predictions(predictions),
        },
        index=test_ids.index,
    )
    cfg = get_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    target = cfg.paths.submission_dir / f"submission_{model_name}_{timestamp}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(target, index=False)
    return target


def validate_submission(
    path: str | Path,
    *,
    expected_ids: pd.Series | None = None,
) -> list[str]:
    """独立校验已存在的提交 CSV。

    参数：path 为 CSV 路径；expected_ids 可选，用于校验 PassengerId 是否完全同序。
    返回：问题列表；空列表表示文件格式、行数、标签值及可选 ID 对齐均合法。
    """
    source = Path(path)
    if not source.exists():
        return [f"提交文件不存在：{source}"]

    try:
        submission = pd.read_csv(
            source,
            dtype={"PassengerId": str, "Transported": str},
        )
    except (OSError, pd.errors.ParserError) as exc:
        return [f"无法读取提交文件 {source}：{exc}"]

    errors: list[str] = []
    if list(submission.columns) != _SUBMISSION_COLUMNS:
        errors.append(
            f"列名及顺序必须为 {_SUBMISSION_COLUMNS}，实际为 {list(submission.columns)}。"
        )
        return errors

    if len(submission) != _TEST_ROW_COUNT:
        errors.append(f"提交行数应为 {_TEST_ROW_COUNT}，实际为 {len(submission)}。")
    if submission["PassengerId"].isna().any() or submission["PassengerId"].duplicated().any():
        errors.append("PassengerId 不得缺失或重复。")
    invalid_values = ~submission["Transported"].isin(["True", "False"])
    if invalid_values.any():
        errors.append("Transported 必须只包含首字母大写的 True 或 False。")

    if expected_ids is not None:
        expected = expected_ids.astype(str).reset_index(drop=True)
        actual = submission["PassengerId"].astype(str).reset_index(drop=True)
        if not actual.equals(expected):
            errors.append("PassengerId 未与期望 ID 完全一致或顺序不一致。")
    return errors
