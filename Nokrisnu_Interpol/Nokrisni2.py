import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import gstools as gs
import os


# ============================================================
# GAMS INDEX FUNCTION
# ============================================================
def compute_gams(P, A):
    A = np.where(A == 0, 0.0001, A)
    numerator = P - ((900 - A)/100) * (P/10)
    ratio = numerator / A
    return np.arctan(1 / ratio)    # arcctg(x)


# ============================================================
# LOOCV FOR NOKRIŠŅI — ExtDrift Kriging
# ============================================================
def Kriging_LOOCV_Nokrisni_Config(
        model_test=1,
        debug=False,
        df_name=r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\Nokrisnu_Interpol\Nokrisni_2020_no_stacijam.csv",
        df_grid_name=r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\Nokrisnu_Interpol\1x1_LV_grid_2024_xy2.csv",
        x="x",
        y="y",
        stat_name="Stacijas"
    ):

    print("\n=== LOOCV NOKRIŠŅI — GAMS + x2 + y2 + elevation(h10) ===\n")

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = pd.read_csv(df_name)
    df_grid = pd.read_csv(df_grid_name)

    # -------------------------
    # ADD DRIFTS (STATIONS)
    # -------------------------
    df["x2"] = df[x]**2
    df["y2"] = df[y]**2
    df["gams"] = compute_gams(df["Nokrisni_mm_lv"], df["elevation"])

    # -------------------------
    # ADD DRIFTS (GRID)
    # -------------------------
    df_grid["x2"] = df_grid[x]**2
    df_grid["y2"] = df_grid[y]**2
    df_grid["elevation"] = df_grid["h10"]     # <--- elevation = h10
    df_grid["gams"] = df["gams"].mean()       # <--- FIXED: mean GAMS, NOT zero!

    # ============================================================
    # CONFIG (same as temperature configs)
    # ============================================================
    configs = {
        1: {
            "class": gs.krige.ExtDrift,
            "param": "ext_drift",
            "drift_cols": ["x2", "y2", "elevation", "gams"],  # TRAIN
            "test_cols":  ["x2", "y2", "elevation", "gams"]   # GRID
        }
    }

    if model_test not in configs:
        raise ValueError("Konfigurācija nav pieejama.")

    cfg = configs[model_test]
    results = []

    # ============================================================
    # LOOCV LOOP
    # ============================================================
    for i in range(len(df)):

        df_test = df.iloc[i]
        df_train = df.drop(index=i)

        # coordinates
        cond_pos = np.vstack([df_train[x], df_train[y]])
        cond_val = df_train["Nokrisni_mm_lv"].to_numpy()

        # drifts
        drift_train = [df_train[col].to_numpy() for col in cfg["drift_cols"]]
        drift_grid  = [df_grid[col].to_numpy()  for col in cfg["test_cols"]]

        # --------------------------------------------------------
        # VARIOGRAM
        # --------------------------------------------------------
        bin_center, gamma = gs.vario_estimate_unstructured(cond_pos, cond_val)

        model = gs.Exponential(
            dim=2,
            var=np.var(cond_val),
            len_scale=41500,   # LVĢMC noteiktais range nokrišņiem
            nugget=0
        )

        # --------------------------------------------------------
        # KRIGING
        # --------------------------------------------------------
        kr = cfg["class"](
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            exact=True,
            **{cfg["param"]: drift_train}
        )

        est_grid, var_grid = kr(
            np.vstack([df_grid[x], df_grid[y]]),
            **{cfg["param"]: drift_grid}
        )

        # find prediction at test point
        tree = cKDTree(df_grid[[x, y]].to_numpy())
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
# RUN FULL LOOCV + SAVE RESULTS
# ============================================================
OUT_DIR = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\LOOCV_NOKRISNI"
os.makedirs(OUT_DIR, exist_ok=True)

df_out = Kriging_LOOCV_Nokrisni_Config(model_test=1)

MAE = df_out["abs_error"].mean()
RMSE = np.sqrt(df_out["error_squared"].mean())
BIAS = df_out["error"].mean()

df_out.to_csv(
    os.path.join(OUT_DIR, "LOOCV_nokrisni2.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n=== RESULTS ===")
print("MAE :", MAE)
print("RMSE:", RMSE)
print("BIAS:", BIAS)
print("\n✓ Pabeigts — dati saglabāti mapē:", OUT_DIR)
