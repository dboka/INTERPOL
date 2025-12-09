import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import gstools as gs
import os


# ============================================================
# GAMS INDEX FUNCTION (LVĢMC MIŠALĒ KOREKCIJA)
# ============================================================
def gams_index(P, A):
    A = np.where(A == 0, 0.0001, A)
    correction = ((900 - A) / 100) * (P / 10)
    X = (P - correction) / A
    X = np.where(X <= 0, np.nan, X)
    return np.arctan(1 / X)


# ============================================================
# ATTACH H5 FROM GRID TO STATIONS
# ============================================================
def attach_grid_h5(df_stac, df_grid):
    tree = cKDTree(df_grid[["x", "y"]].values)
    _, idx = tree.query(df_stac[["x", "y"]].values, k=1)
    df_stac["h5"] = df_grid.iloc[idx]["h5"].values
    return df_stac


# ============================================================
# LOOCV — LVĢMC DRIFT MODEL (x, y, h5, gams)
# ============================================================
def Kriging_LOOCV_Nokrisni_Config2(
        df_name=r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\Nokrisnu_Interpol\Nokrisni_2020_no_stacijam.csv",
        df_grid_name=r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\Nokrisnu_Interpol\1x1_LV_grid_2024_xy2.csv",
        x="x",
        y="y",
        stat_name="Stacijas"
    ):

    print("\n=== LOOCV NOKRIŠŅI 2 — x + y + h5 + Gams (LVĢMC) ===\n")

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = pd.read_csv(df_name)
    df_grid = pd.read_csv(df_grid_name)

    # -------------------------
    # GAMS STATIONS
    # -------------------------
    df["Gams"] = gams_index(df["Nokrisni_mm_lv"], df["elevation"])

    # -------------------------
    # ADD H5 TO STATIONS
    # -------------------------
    df = attach_grid_h5(df, df_grid)

    # -------------------------
    # GRID GAMS = MEAN GAMS
    # -------------------------
    df_grid["Gams"] = df["Gams"].mean()

    drift_cols = ["x", "y", "h5", "Gams"]

    results = []

    # ============================================================
    # LOOCV LOOP
    # ============================================================
    for i in range(len(df)):

        df_test = df.iloc[i]
        df_train = df.drop(index=i)

        cond_pos = np.vstack([df_train[x], df_train[y]])
        cond_val = df_train["Nokrisni_mm_lv"].values

        drift_train = [df_train[col].values for col in drift_cols]

        drift_grid = [
            df_grid["x"].values,
            df_grid["y"].values,
            df_grid["h5"].values,
            df_grid["Gams"].values
        ]

        # --------------------------------------------------------
        # VARIOGRAM (FIXED — LVĢMC METHOD)
        # --------------------------------------------------------
        model = gs.Exponential(
            dim=2,
            var=np.var(cond_val),
            len_scale=41500,
            nugget=0
        )

        # --------------------------------------------------------
        # KRIGING WITH DRIFT
        # --------------------------------------------------------
        kr = gs.krige.ExtDrift(
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            ext_drift=drift_train,
            exact=True
        )

        est_grid, _ = kr(
            np.vstack([df_grid[x], df_grid[y]]),
            ext_drift=drift_grid
        )


        # nearest grid point
        tree = cKDTree(df_grid[[x, y]].values)
        _, idx = tree.query([[df_test[x], df_test[y]]], k=1)

        pred = est_grid[idx][0]
        true = df_test["Nokrisni_mm_lv"]
        err = pred - true

        results.append({
            "station": df_test[stat_name],
            "true": true,
            "pred": pred,
            "error": err,
            "abs_error": abs(err),
            "error_squared": err**2
        })

    return pd.DataFrame(results)



# ============================================================
# RUN + SAVE RESULTS
# ============================================================
OUT_DIR = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\LOOCV_NOKRISNI"
os.makedirs(OUT_DIR, exist_ok=True)

df_out = Kriging_LOOCV_Nokrisni_Config2()

MAE = df_out["abs_error"].mean()
RMSE = np.sqrt(df_out["error_squared"].mean())
BIAS = df_out["error"].mean()

df_out.to_csv(
    os.path.join(OUT_DIR, "LOOCV_nokrisni3.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n=== RESULTS (LVĢMC Model) ===")
print("MAE :", MAE)
print("RMSE:", RMSE)
print("BIAS:", BIAS)
print("\n✓ Saglabāts kā: LOOCV_nokrisni2.csv iekš:", OUT_DIR)
