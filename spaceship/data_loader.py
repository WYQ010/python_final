# -*- coding: utf-8 -*-
"""统一读数入口（《统一接口定义》§3.1，负责人 A）。

只负责读取与 dtype 规范（见接口 §2.3 表），不做任何清洗、填补、加列、删行、去重；
清洗与缺失处理是 preprocessing 的职责。
"""
from __future__ import annotations

import pandas as pd

from spaceship.config import get_config

_TRAIN_COLUMNS = [
    "PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name", "Transported",
]
_TEST_COLUMNS = _TRAIN_COLUMNS[:-1]
_SAMPLE_COLUMNS = ["PassengerId", "Transported"]
_NUMERIC_COLUMNS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
_TEXT_COLUMNS = ["HomePlanet", "Destination", "Cabin", "Name"]
_BOOLEAN_INPUT_COLUMNS = ["CryoSleep", "VIP"]


def _load_csv(
    filename: str,
    expected_columns: list[str],
    expected_rows: int,
) -> pd.DataFrame:
    """读取 CSV 并校验文件名、形状、列顺序和 PassengerId 唯一性。"""
    path = get_config().paths.data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件，期望路径：{path}。"
            f"请按 README「快速开始」第 3 步获取比赛数据，"
            f"并将 {filename} 放入 data/ 目录（文件名不变）。"
        )
    # PassengerId 形如 0001_01，显式按字符串读取防前导零问题
    frame = pd.read_csv(path, dtype={"PassengerId": str})
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"数据文件 {path} 的列名或顺序不正确；"
            f"期望 {expected_columns}，实际 {list(frame.columns)}。"
        )
    if len(frame) != expected_rows:
        raise ValueError(
            f"数据文件 {path} 的行数不正确；期望 {expected_rows}，实际 {len(frame)}。"
        )
    if frame["PassengerId"].isna().any() or frame["PassengerId"].duplicated().any():
        raise ValueError(f"数据文件 {path} 的 PassengerId 不得缺失或重复。")
    return frame


def _normalise_train_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """仅规范训练集读数 dtype，不进行任何清洗、填补或特征衍生。"""
    out = frame.copy()
    try:
        out[_NUMERIC_COLUMNS] = out[_NUMERIC_COLUMNS].apply(
            pd.to_numeric,
            errors="raise",
        ).astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("训练集的 Age 或消费列必须是可解析的数值。") from exc
    out[_TEXT_COLUMNS] = out[_TEXT_COLUMNS].astype("object")
    out[_BOOLEAN_INPUT_COLUMNS] = out[_BOOLEAN_INPUT_COLUMNS].astype("object")
    labels = out["Transported"].astype(str)
    if not labels.isin(["True", "False"]).all():
        raise ValueError("训练集 Transported 必须只包含 True 或 False，且不得缺失。")
    out["Transported"] = labels.eq("True")
    return out


def load_train() -> pd.DataFrame:
    """读取训练集 train.csv：(8693, 14)，含目标列 Transported（bool）。"""
    return _normalise_train_dtypes(_load_csv("train.csv", _TRAIN_COLUMNS, 8693))


def load_test() -> pd.DataFrame:
    """读取测试集 test.csv：(4277, 13)，无目标列。"""
    frame = _load_csv("test.csv", _TEST_COLUMNS, 4277)
    try:
        frame[_NUMERIC_COLUMNS] = frame[_NUMERIC_COLUMNS].apply(
            pd.to_numeric,
            errors="raise",
        ).astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("测试集的 Age 或消费列必须是可解析的数值。") from exc
    frame[_TEXT_COLUMNS] = frame[_TEXT_COLUMNS].astype("object")
    frame[_BOOLEAN_INPUT_COLUMNS] = frame[_BOOLEAN_INPUT_COLUMNS].astype("object")
    return frame


def load_sample_submission() -> pd.DataFrame:
    """读取提交模板 sample_submission.csv：(4277, 2)，列 = [PassengerId, Transported]。"""
    return _load_csv("sample_submission.csv", _SAMPLE_COLUMNS, 4277)
