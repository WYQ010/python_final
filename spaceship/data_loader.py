    # -*- coding: utf-8 -*-
"""统一读数入口（《统一接口定义》§3.1，负责人 A）。

只负责读取与 dtype 规范（见接口 §2.3 表），不做任何清洗、填补、加列、删行、去重；
清洗与缺失处理是 preprocessing 的职责。
"""
from __future__ import annotations

import pandas as pd

from spaceship.config import get_config


def _load_csv(filename: str) -> pd.DataFrame:
    """按 config 路径读取单个 CSV；文件缺失时抛含期望路径的 FileNotFoundError。"""
    path = get_config().paths.data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件，期望路径：{path}。"
            f"请按 README「快速开始」第 3 步获取比赛数据，"
            f"并将 {filename} 放入 data/ 目录（文件名不变）。"
        )
    # PassengerId 形如 0001_01，显式按字符串读取防前导零问题
    return pd.read_csv(path, dtype={"PassengerId": str})


def load_train() -> pd.DataFrame:
    """读取训练集 train.csv：(8693, 14)，含目标列 Transported（bool）。"""
    return _load_csv("train.csv")


def load_test() -> pd.DataFrame:
    """读取测试集 test.csv：(4277, 13)，无目标列。"""
    return _load_csv("test.csv")


def load_sample_submission() -> pd.DataFrame:
    """读取提交模板 sample_submission.csv：(4277, 2)，列 = [PassengerId, Transported]。"""
    return _load_csv("sample_submission.csv")
