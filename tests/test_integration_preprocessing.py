# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from spaceship.preprocessing import preprocess
from tests.data_factory import make_dummy_raw_data

@pytest.fixture
def raw_df():
    return make_dummy_raw_data(n_rows=20)

def test_preprocess_shape_and_ids_unchanged(raw_df):
    out = preprocess(raw_df)
    assert out.shape == raw_df.shape
    assert out.index.equals(raw_df.index)
    assert list(out.columns) == list(raw_df.columns)
    assert out["PassengerId"].equals(raw_df["PassengerId"])
    assert out["Transported"].equals(raw_df["Transported"])

def test_preprocess_no_missing_values(raw_df):
    out = preprocess(raw_df)
    assert out.isna().sum().sum() == 0

def test_preprocess_bool_dtypes(raw_df):
    out = preprocess(raw_df)
    assert out["CryoSleep"].dtype == bool
    assert out["VIP"].dtype == bool

def test_preprocess_cryosleep_spend_linkage(raw_df):
    out = preprocess(raw_df)
    cryo_true_rows = out["CryoSleep"] == True
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    
    assert (out.loc[cryo_true_rows, spend_cols] == 0).all().all()

def test_preprocess_value_ranges(raw_df):
    out = preprocess(raw_df)
    assert (out["Age"] >= 0).all() and (out["Age"] <= 79).all()
    
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        assert (out[col] >= 0).all()