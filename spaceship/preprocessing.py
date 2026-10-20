# -*- coding: utf-8 -*-
"""清洗与缺失填补（《统一接口定义》§3.2，负责人 A）。

策略一律读取 config.preprocess；本模块不做特征衍生、不改行数/索引/列集、
不动 PassengerId 与 Transported。契约：输出全表无缺失，CryoSleep/VIP 为纯 bool，
原 CryoSleep==True 行的 5 项消费填补后均为 0。
"""
from __future__ import annotations

import pandas as pd

from spaceship.config import get_config

_SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _fill_mode(s: pd.Series) -> pd.Series:
    """众数填补（无任何非缺失值时原样返回，属空列兜底）。"""
    if s.notna().any():
        return s.fillna(s.mode().iloc[0])
    return s


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """按 config 策略填补缺失，返回全新 DataFrame。

    参数 df：load_train() / load_test() 的输出（有无 Transported 列均可，原样透传）。
    返回：同 shape / 索引 / 列名；全表无缺失；CryoSleep / VIP 为纯 bool。
    """
    cfg = get_config().preprocess
    out = df.copy()

    # 1) CryoSleep 缺失：5 项消费全为 0 → True，否则 False（缺失值不算 0）
    cryo_na = out["CryoSleep"].isna()
    if cryo_na.any():
        all_zero = out.loc[cryo_na, _SPEND_COLS].eq(0).all(axis=1)
        out.loc[cryo_na, "CryoSleep"] = all_zero

    # 2) VIP 缺失 → config.vip_fill（默认 False，True 仅约 2%）
    out["VIP"] = out["VIP"].fillna(cfg.vip_fill)

    # 3) 消费列缺失：cryo_then_zero——CryoSleep=True → 0；其余 → 0（众数）
    #    两种分支结果同为 0，保留分支以承载 config 语义与后续策略扩展
    if cfg.spend_fill == "cryo_then_zero":
        out.loc[out["CryoSleep"] == True, _SPEND_COLS] = (
            out.loc[out["CryoSleep"] == True, _SPEND_COLS].fillna(0.0)
        )
        out[_SPEND_COLS] = out[_SPEND_COLS].fillna(0.0)
    else:  # 未知策略名兜底为 0
        out[_SPEND_COLS] = out[_SPEND_COLS].fillna(0.0)

    # 4) Age
    if cfg.age_fill == "median":
        out["Age"] = out["Age"].fillna(out["Age"].median())
    elif cfg.age_fill == "mode":
        out["Age"] = _fill_mode(out["Age"])
    else:  # zero
        out["Age"] = out["Age"].fillna(0.0)

    # 5) 枚举列
    if cfg.homeplanet_fill == "unknown":
        out["HomePlanet"] = out["HomePlanet"].fillna("Unknown")
    else:  # mode
        out["HomePlanet"] = _fill_mode(out["HomePlanet"])
    if cfg.destination_fill == "unknown":
        out["Destination"] = out["Destination"].fillna("Unknown")
    else:  # mode
        out["Destination"] = _fill_mode(out["Destination"])

    # 6) Cabin / Name 占位（Cabin 占位串可被后续 deck/num/side 拆分统一处理）
    out["Cabin"] = out["Cabin"].fillna(cfg.cabin_placeholder)
    out["Name"] = out["Name"].fillna(cfg.name_placeholder)

    # 7) 布尔化（此时两列已无缺失，astype 安全）
    out["CryoSleep"] = out["CryoSleep"].astype(bool)
    out["VIP"] = out["VIP"].astype(bool)

    return out
