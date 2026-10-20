# -*- coding: utf-8 -*-
"""通用工具（结构冻结，《统一接口定义》§6）。"""
from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """固定 Python 与 NumPy 全局随机种子；sklearn 侧一律显式传 random_state。"""
    random.seed(seed)
    np.random.seed(seed)
