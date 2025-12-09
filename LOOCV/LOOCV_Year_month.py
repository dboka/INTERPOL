import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import gstools as gs
import os
from pyproj import Transformer

# ============================================================
# LOOCV — CONFIG 4: elevation + to_sea + continentality + LAT
# ============================================================

def Kriging_LOOCV_Config4(
        dati_ver,
        model_test=4,
        debug=False,
        df_name=r"C:\Users\deniss.boka\Interpol\data\staciju_dati_norma3.csv",
        df_grid_name=r"C:\Users\deniss.boka\Interpol\data\1x1_LV_grid_2024_xy2.csv",
        x='x', y='y',
        z='elevation',
        H='h10',
        to_sea='to_sea',
        kont='cont_tas',
        stat_ind='gh_id',
        stat_name='name'
    ):

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = pd.read_csv(df_name, sep=';')
    df_grid = pd.read_csv(df_grid_name)

    # -------------------------
    # ADD LATITUDE (convert LKS92 → WGS84)
    # -------------------------
    transformer = Transformer.from_crs(3059, 4326, always_xy=True)

    df['lon'], df['lat'] = transformer.transform(df[x].values, df[y].values)
    df_grid['lon'], df_grid['lat'] = transformer.transform(df_grid[x].values, df_grid[y].values)

    # ============================================================
    # 4 DRIFTS
    # ============================================================
    configs = {
        4: {
            "class": gs.krige.ExtDrift,
            "param": "ext_drift",
            "drift_cols": [z, to_sea, kont, 'lat'],   # TRAIN
            "test_cols":  [H, to_sea, kont, 'lat']    # GRID
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

        cond_pos = np.vstack([df_train[x], df_train[y]])
        cond_val = df_train[dati_ver].to_numpy()

        drift_train = [df_train[col].to_numpy() for col in cfg["drift_cols"]]
        drift_grid  = [df_grid[col].to_numpy()  for col in cfg["test_cols"]]

        # -------------------------
        # VARIOGRAM
        # -------------------------
        bin_center, gamma = gs.vario_estimate_unstructured(cond_pos, cond_val)

        model = gs.Gaussian(
            dim=2,
            var=np.var(cond_val),
            len_scale=39500,
            nugget=0
        )
        model.fit_variogram(bin_center, gamma)

        # -------------------------
        # KRIGING
        # -------------------------
        krige = cfg["class"](
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            exact=True,
            **{cfg["param"]: drift_train}
        )

        est_grid, var_grid = krige(
            np.vstack([df_grid[x], df_grid[y]]),
            **{cfg["param"]: drift_grid}
        )

        # -------------------------
        # FIND NEAREST GRID POINT
        # -------------------------
        tree = cKDTree(df_grid[[x, y]].to_numpy())
        _, idx = tree.query([[df_test[x], df_test[y]]], k=1)

        pred_val = est_grid[idx][0]
        pred_var = var_grid[idx][0]
        true_val = df_test[dati_ver]

        error = pred_val - true_val

        results.append({
            "station": df_test[stat_name],
            "id": df_test[stat_ind],
            "month": dati_ver,
            "true": true_val,
            "pred": pred_val,
            "error": error,
            "abs_error": abs(error),
            "error_squared": error**2,
            "variance": pred_var
        })

    return pd.DataFrame(results)




# ============================================================
# LOOCV FOR ALL MONTHS
# ============================================================

OUT_DIR = r"C:\Users\deniss.boka\Interpol\results\LOOCV_Gorcinskis_LAT"
os.makedirs(OUT_DIR, exist_ok=True)

months = [
    "Janvāris","Februāris","Marts","Aprīlis","Maijs","Jūnijs",
    "Jūlijs","Augusts","Septembris","Oktobris","Novembris","Decembris"
]

summary = []
H_VALUE = "h10"

print(f"\n=== CONFIG 4 — elevation + to_sea + cont_tas + LATITUDE — H = {H_VALUE} ===\n")

for month in months:

    print(f"--- LOOCV {month} ---")

    df_res = Kriging_LOOCV_Config4(month, model_test=4, H=H_VALUE)

    MAE = df_res["abs_error"].mean()
    RMSE = np.sqrt(df_res["error_squared"].mean())
    BIAS = df_res["error"].mean()

    summary.append([month, MAE, RMSE, BIAS])

    df_res.to_csv(
        os.path.join(OUT_DIR, f"LOOCV_{month}_{H_VALUE}_LAT.csv"),
        index=False,
        encoding="utf-8-sig"
    )

df_summary = pd.DataFrame(summary, columns=["Month","MAE","RMSE","Bias"])
df_summary.to_csv(
    os.path.join(OUT_DIR, f"LOOCV_summary_{H_VALUE}_LAT.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n✓ PABEIGTS — visi LOOCV ar LATITUDE saglabāti mapē:")
print(OUT_DIR)
