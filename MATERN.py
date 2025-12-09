import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import gstools as gs
from pyproj import Transformer
import os


# ============================================================
# NORMALIZER
# ============================================================
def normalize_numpy(train_arr, test_arr):
    mean = np.mean(train_arr, axis=0)
    std = np.std(train_arr, axis=0)
    std = np.where(std == 0, 1, std)
    return (train_arr - mean) / std, (test_arr - mean) / std


# ============================================================
# SIMULATED PoweredExponential (fallback)
# ============================================================
def poweredexp_semivar(h, var, len_scale, alpha):
    return var * (1 - np.exp(-(h / len_scale)))**alpha


# ============================================================
# VARIOGRAM MODEL SELECTOR
# ============================================================
def build_variogram_model(model_type, var_val, len_scale, nugget_val):
    model_type = model_type.lower()

    if model_type == "gaussian":
        return gs.Gaussian(dim=2, var=var_val, len_scale=len_scale, nugget=nugget_val)

    if model_type == "matern":
        return gs.Matern(dim=2, var=var_val, len_scale=len_scale, nu=1.0, nugget=nugget_val)

    if model_type == "powerexp":
        if hasattr(gs, "PowerExp"):
            return gs.PowerExp(dim=2, var=var_val, len_scale=len_scale, alpha=1.3, nugget=nugget_val)
        else:
            print("⚠ PowerExp not available → fallback")
            return None

    raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# DAILY LOOCV WITH CONFIG4 (4 DRIFTS)
# ============================================================
def LOOCV_one_day_CONFIG4(df_day, df_grid, model_type):

    results = []

    drift_cols_train = ["elevation", "to_sea", "cont_tas", "lat84"]
    drift_cols_test  = ["h10", "to_sea", "cont_tas", "lat84"]

    grid_pos = np.vstack([df_grid["x"], df_grid["y"]])
    drift_grid_raw = df_grid[drift_cols_test].to_numpy()

    tree = cKDTree(df_grid[["x", "y"]].to_numpy())

    for i in range(len(df_day)):

        df_test = df_day.iloc[i]
        df_train = df_day.drop(index=df_test.name)

        cond_pos = np.vstack([df_train["x"], df_train["y"]])
        cond_val = df_train["value"].to_numpy()

        drift_train_raw = df_train[drift_cols_train].to_numpy()

        drift_train_norm, drift_grid_norm = normalize_numpy(
            drift_train_raw, drift_grid_raw
        )

        var_val = max(np.var(cond_val), 0.01)
        nugget_val = 0.05 * var_val
        len_scale = 39500

        bin_center, gamma = gs.vario_estimate_unstructured(cond_pos, cond_val)

        model = build_variogram_model(model_type, var_val, len_scale, nugget_val)

        if model is None:
            model = gs.Exponential(dim=2, var=var_val, len_scale=len_scale, nugget=nugget_val)
        else:
            try:
                model.fit_variogram(bin_center, gamma, nugget=nugget_val)
            except:
                pass

        krige = gs.krige.ExtDrift(
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            exact=True,
            ext_drift=[drift_train_norm[:, j] for j in range(drift_train_norm.shape[1])]
        )

        est_grid, var_grid = krige(
            grid_pos,
            ext_drift=[drift_grid_norm[:, j] for j in range(drift_grid_norm.shape[1])]
        )

        _, idx = tree.query([[df_test["x"], df_test["y"]]], k=1)

        pred = est_grid[idx][0]
        true = df_test["value"]
        error = pred - true

        results.append({
            "date": df_test["date"].strftime("%Y-%m-%d"),
            "station": df_test["name"],
            "true": true,
            "pred": pred,
            "error": error,
            "abs_error": abs(error)
        })

    return results


# ============================================================
# MAIN DAILY LOOP
# ============================================================
df = pd.read_csv(r"C:\Users\deniss.boka\Interpol\CSV pa gadiem\2024_long.csv", sep=";")
df["date"] = pd.to_datetime(dict(year=df.Gads, month=df.Menesis, day=df.Diena))

df_grid = pd.read_csv(r"C:\Users\deniss.boka\Interpol\data\1x1_LV_grid_2024_xy2.csv")

transformer = Transformer.from_crs(3059, 4326, always_xy=True)
df["lon"], df["lat"] = transformer.transform(df["x"], df["y"])
df_grid["lon"], df_grid["lat"] = transformer.transform(df_grid["x"], df_grid["y"])

df["lat84"] = df["lat"]
df_grid["lat84"] = df_grid["lat"]

df["h10"] = df["elevation"]  # unify


# ============================================================
# RUN DAILY LOOCV
# ============================================================
MODEL = "matern"  # "gaussian", "matern", "powerexp"

all_results = []

unique_dates = df["date"].unique()

for d in unique_dates:
    
    df_day = df[df["date"] == d]
    if len(df_day) < 5:
        continue

    print("DAY:", d.strftime("%Y-%m-%d"))

    day_res = LOOCV_one_day_CONFIG4(df_day, df_grid, MODEL)
    day_df = pd.DataFrame(day_res)

    all_results.append(day_df)

out_df = pd.concat(all_results, ignore_index=True)

OUT = fr"C:\Users\deniss.boka\Interpol\data2024\LOOCV_daily_2024_CONFIG4_{MODEL}.csv"
out_df.to_csv(OUT, sep=";", index=False, encoding="utf-8-sig")

print("\n✓ SAVED:", OUT)
