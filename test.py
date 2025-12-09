import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import gstools as gs
from pyproj import Transformer
import os

# ============================================
# LOAD DAILY LONG FORMAT DATA
# ============================================

df = pd.read_csv(
    r"C:\Users\deniss.boka\Interpol\CSV pa gadiem\2024_long.csv",
    sep=";"
)

df["date"] = pd.to_datetime(dict(year=df.Gads, month=df.Menesis, day=df.Diena))

# ============================================
# LOAD GRID
# ============================================
df_grid = pd.read_csv(
    r"C:\Users\deniss.boka\Interpol\data\1x1_LV_grid_2024_xy2.csv"
)

# ============================================
# COORD TRANSFORM FOR LAT/LON (not used as drift)
# ============================================
transformer = Transformer.from_crs(3059, 4326, always_xy=True)
df["lon"], df["lat"] = transformer.transform(df["x"], df["y"])
df_grid["lon"], df_grid["lat"] = transformer.transform(df_grid["x"], df_grid["y"])

df["lat84"] = df["lat"]
df_grid["lat84"] = df_grid["lat"]

# ============================================
# DAILY MODEL — ONLY elevation + to_sea (2 DRIFTS)
# ============================================

def LOOCV_one_day_MODEL2(df_day):

    results = []

    # TRAIN DRIFTS (ONLY TWO!)
    drift_cols_train = ["elevation", "to_sea"]

    # TEST DRIFTS (GRID)
    drift_cols_test = ["h10", "to_sea"]

    # Precompute grid drift values
    drift_grid = [df_grid[col].to_numpy() for col in drift_cols_test]
    grid_pos = np.vstack([df_grid["x"], df_grid["y"]])

    tree = cKDTree(df_grid[["x", "y"]].to_numpy())

    for i in range(len(df_day)):

        df_test = df_day.iloc[i]
        df_train = df_day.drop(index=df_test.name)

        cond_pos = np.vstack([df_train["x"], df_train["y"]])
        cond_val = df_train["value"].to_numpy()

        # Drift for stations
        drift_train = [df_train[c].to_numpy() for c in drift_cols_train]

        # Variogram
        bin_center, gamma = gs.vario_estimate_unstructured(cond_pos, cond_val)

        model = gs.Gaussian(
            dim=2,
            var=np.var(cond_val),
            len_scale=39500,
            nugget=0
        )
        model.fit_variogram(bin_center, gamma)

        # Kriging
        krige = gs.krige.ExtDrift(
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            exact=True,
            ext_drift=drift_train
        )

        est_grid, var_grid = krige(
            grid_pos,
            ext_drift=drift_grid
        )

        # Nearest grid pixel
        _, idx = tree.query([[df_test["x"], df_test["y"]]], k=1)

        pred_val = est_grid[idx][0]
        pred_var = var_grid[idx][0]
        true_val = df_test["value"]

        results.append({
            "date": df_test["date"].strftime("%Y-%m-%d"),
            "station": df_test["name"],
            "id": df_test["gh_id"],
            "true": true_val,
            "pred": pred_val,
            "error": pred_val - true_val,
            "abs_error": abs(pred_val - true_val),
            "variance": pred_var
        })

    return results


# ============================================
# RUN LOOCV FOR ENTIRE YEAR
# ============================================

all_results = []
unique_dates = df["date"].unique()

for d in unique_dates:

    df_day = df[df["date"] == d]

    if len(df_day) < 5:
        continue

    print("DAY:", d.strftime("%Y-%m-%d"))

    day_results = LOOCV_one_day_MODEL2(df_day)
    day_df = pd.DataFrame(day_results)

    # Metrics
    MAE = day_df["abs_error"].mean()
    RMSE = np.sqrt((day_df["error"] ** 2).mean())
    BIAS = day_df["error"].mean()

    day_df["MAE"] = MAE
    day_df["RMSE"] = RMSE
    day_df["BIAS"] = BIAS

    all_results.extend(day_df.to_dict("records"))

# ============================================
# SAVE OUTPUT
# ============================================

OUT = r"C:\Users\deniss.boka\Interpol\data2024\LOOCV_daily_2024_Model2_FULL.csv"
pd.DataFrame(all_results).to_csv(OUT, sep=";", index=False, encoding="utf-8-sig")

print("\n✓ DAILY MODEL 2 (elevation + to_sea) SAVED:")
print(OUT)