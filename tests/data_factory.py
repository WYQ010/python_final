import pandas as pd
import numpy as np

def make_dummy_raw_data(n_rows: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        "PassengerId": [f"{i:04d}_01" for i in range(1, n_rows + 1)],
        "HomePlanet": np.random.choice(["Earth", "Europa", "Mars", None], n_rows),
        "CryoSleep": np.random.choice([True, False, None], n_rows),
        "Cabin": np.random.choice(["A/1/P", "T/5/S", None], n_rows),
        "Destination": np.random.choice(["TRAPPIST-1e", "55 Cancri e", None], n_rows),
        "Age": np.random.choice([0.0, 18.0, 79.0, np.nan], n_rows),
        "VIP": np.random.choice([True, False, None], n_rows),
        "RoomService": np.random.choice([0.0, 100.0, np.nan], n_rows),
        "FoodCourt": np.random.choice([0.0, 200.0, np.nan], n_rows),
        "ShoppingMall": np.random.choice([0.0, 300.0, np.nan], n_rows),
        "Spa": np.random.choice([0.0, 400.0, np.nan], n_rows),
        "VRDeck": np.random.choice([0.0, 500.0, np.nan], n_rows),
        "Name": np.random.choice(["A A", None], n_rows),
        "Transported": np.random.choice([True, False], n_rows)
    })

    # 严格遵循 T1 要求：所有 CryoSleep 为 True 的行，5 项消费必须全为 0.0
    cryo_true = df["CryoSleep"] == True
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df.loc[cryo_true, spend_cols] = 0.0

    if n_rows > 1:
        df.loc[1, "PassengerId"] = "0001_02"

    return df