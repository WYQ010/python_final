# -*- coding: utf-8 -*-
"""特征衍生与编码（《统一接口定义》§3.3，负责人 A）——A/B 交接面。

train+test 联合处理（组统计、类别编码需全局一致）；只做无监督聚合，
不使用 Transported、不引入 test 独有信息（防泄漏）。输出契约六条见接口文档。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spaceship.data_loader import load_sample_submission

_SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
_CAT_COLS = ["HomePlanet", "Destination", "deck", "side"]
_NUM_COLS = ["Age", "is_minor", "CryoSleep", "VIP", *_SPEND_COLS,
             "spend_total", "spend_zero_cnt", "cabin_num", "group_size", "in_group"]


@dataclass(frozen=True)
class FeatureSet:
    """特征契约载体（接口 §3.3）：X_train/y_train/X_test/test_ids。"""

    X_train: pd.DataFrame
    y_train: pd.Series      # dtype bool
    X_test: pd.DataFrame    # 与 X_train 完全同列同序
    test_ids: pd.Series     # dtype str，与 sample_submission 同序


def _derive_base(df: pd.DataFrame) -> pd.DataFrame:
    """在单份 clean 数据上派生行级特征（不含跨行统计）。"""
    out = df.copy()
    # Cabin → deck / num / side（缺失占位 U/-1/U 可被统一拆分）
    cabin = out["Cabin"].astype(str).str.split("/", expand=True)
    out["deck"] = cabin[0]
    out["cabin_num"] = pd.to_numeric(cabin[1], errors="coerce").fillna(-1).astype("int64")
    out["side"] = cabin[2]
    # 消费聚合
    out["spend_total"] = out[_SPEND_COLS].sum(axis=1)
    out["spend_zero_cnt"] = out[_SPEND_COLS].eq(0).sum(axis=1).astype("int64")
    # 年龄衍生
    out["is_minor"] = out["Age"] < 13
    # 组号（组规模等跨行统计在合并后计算）
    out["group"] = out["PassengerId"].astype(str).str.split("_").str[0]
    return out


def build_features(train_clean: pd.DataFrame, test_clean: pd.DataFrame) -> FeatureSet:
    """联合构造特征矩阵；契约六条（行数/同列同序/无缺失全数值/y 一致/ids 同序/确定性）。"""
    tr = _derive_base(train_clean)
    te = _derive_base(test_clean)

    # 跨行统计：组规模按 train+test 合并计算（无监督，不含目标信息，无泄漏）
    group_counts = pd.concat([tr["group"], te["group"]], ignore_index=True).value_counts()
    for d in (tr, te):
        d["group_size"] = d["group"].map(group_counts).astype("int64")
        d["in_group"] = d["group_size"] > 1

    # 类别列联合独热（保证 train/test 同列同序、无未见类别）
    combined = pd.concat([tr, te], keys=["train", "test"], sort=False)
    dummies = pd.get_dummies(combined[_CAT_COLS], columns=_CAT_COLS, dtype=bool)
    X = pd.concat(
        [combined[_NUM_COLS].reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )
    X_tr = X.iloc[: len(tr)].reset_index(drop=True)
    X_te = X.iloc[len(tr):].reset_index(drop=True)

    # 契约⑤：test_ids 与 sample_submission 完全一致且同序（ PassengerId 唯一，按位重排）
    sample_ids = load_sample_submission()["PassengerId"].astype(str)
    te_ids = te["PassengerId"].astype(str).reset_index(drop=True)
    pos = {pid: i for i, pid in enumerate(te_ids)}
    order = [pos[pid] for pid in sample_ids]      # 缺 id 会 KeyError，契约违规即失败
    X_te = X_te.iloc[order].reset_index(drop=True)
    test_ids = sample_ids.reset_index(drop=True).rename("PassengerId")

    y_train = train_clean["Transported"].astype(bool).reset_index(drop=True).rename("Transported")
    return FeatureSet(X_train=X_tr, y_train=y_train, X_test=X_te, test_ids=test_ids)
