import pandas as pd
import numpy as np
import gstools as gs
from pyproj import Transformer
from scipy.spatial import cKDTree
from qgis.core import QgsProject, QgsField, QgsFeature, QgsVectorLayer, QgsGeometry, QgsPointXY
from PyQt5.QtCore import QVariant


# =========================================================
# INPUT SETTINGS — CHOOSE DAY
# =========================================================
MENESIS = 1   # <-- Month number (1–12)
DIENA = 4     # <-- Day number (1–31)
MODEL = "matern"   # "gaussian", "matern", "powerexp"

stations_csv = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\CSV pa gadiem\2024_long.csv"
grid_csv     = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\1x1_LV_grid_2024_xy2.csv"


# =========================================================
# NORMALIZER
# =========================================================
def normalize_numpy(train_arr, test_arr):
    mean = np.mean(train_arr, axis=0)
    std = np.std(train_arr, axis=0)
    std = np.where(std == 0, 1, std)
    return (train_arr - mean) / std, (test_arr - mean) / std


# =========================================================
# VARIOGRAM MODEL SELECTOR
# =========================================================
def select_model(model_type, var_val, len_scale, nugget):
    if model_type == "gaussian":
        return gs.Gaussian(dim=2, var=var_val, len_scale=len_scale, nugget=nugget)

    elif model_type == "matern":
        return gs.Matern(dim=2, var=var_val, len_scale=len_scale, nu=1.0, nugget=nugget)

    elif model_type == "powerexp":
        if hasattr(gs, "PowerExp"):
            return gs.PowerExp(dim=2, var=var_val, len_scale=len_scale, alpha=1.3, nugget=nugget)
        else:
            print("⚠ PowerExp not found — using Exponential fallback")
            return gs.Exponential(dim=2, var=var_val, len_scale=len_scale, nugget=nugget)

    else:
        raise ValueError("Unknown model type:", model_type)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(stations_csv, sep=";")
df_day = df[(df["Menesis"] == MENESIS) & (df["Diena"] == DIENA)]

if df_day.empty:
    print("❌ NAV DATU ŠAI DIENAI:", MENESIS, DIENA)
    raise SystemExit

print("✓ Loaded STATIONS:", len(df_day))


# =========================================================
# LAT84 TRANSFORM
# =========================================================
transformer = Transformer.from_crs(3059, 4326, always_xy=True)

df_day["lon84"], df_day["lat84"] = transformer.transform(
    df_day["x"].to_numpy(), df_day["y"].to_numpy()
)


# =========================================================
# TRAIN POS + VALUE
# =========================================================
cond_pos = np.vstack([df_day["x"], df_day["y"]])
cond_val = df_day["value"].to_numpy()

# RAW DRIFTS
train_raw = df_day[["elevation", "to_sea", "cont_tas", "lat84"]].to_numpy()


# =========================================================
# LOAD GRID
# =========================================================
df_grid = pd.read_csv(grid_csv)
df_grid["lon84"], df_grid["lat84"] = transformer.transform(
    df_grid["x"].to_numpy(), df_grid["y"].to_numpy()
)

grid_pos = np.vstack([df_grid["x"], df_grid["y"]])

test_raw = df_grid[["h10", "to_sea", "cont_tas", "lat84"]].to_numpy()

print("✓ GRID SIZE:", len(df_grid))


# =========================================================
# NORMALIZATION (NEW LOOCV LOGIC)
# =========================================================
train_norm, test_norm = normalize_numpy(train_raw, test_raw)


# =========================================================
# VARIOGRAM
# =========================================================
var_val = max(np.var(cond_val), 0.01)
nugget = 0.05 * var_val
len_scale = 39500

bin_center, gamma = gs.vario_estimate_unstructured(cond_pos, cond_val)

model = select_model(MODEL, var_val, len_scale, nugget)

try:
    model.fit_variogram(bin_center, gamma, nugget=nugget)
except:
    print("⚠ Variogram fit failed — using initial parameters")


# =========================================================
# EXT DRIFT KRIGING
# =========================================================
krige = gs.krige.ExtDrift(
    model,
    cond_pos=cond_pos,
    cond_val=cond_val,
    exact=True,
    ext_drift=[train_norm[:, j] for j in range(train_norm.shape[1])]
)

est, var = krige(
    grid_pos,
    ext_drift=[test_norm[:, j] for j in range(test_norm.shape[1])]
)


# =========================================================
# CREATE QGIS LAYER (RESULTS)
# =========================================================
layer = QgsVectorLayer("Point?crs=EPSG:3059", f"Kriging_{MENESIS:02d}_{DIENA:02d}_{MODEL}", "memory")
pr = layer.dataProvider()

pr.addAttributes([
    QgsField("x", QVariant.Double),
    QgsField("y", QVariant.Double),
    QgsField("est", QVariant.Double),
    QgsField("var", QVariant.Double)
])
layer.updateFields()

features = []
gx = df_grid["x"].to_numpy()
gy = df_grid["y"].to_numpy()

for i in range(len(gx)):
    f = QgsFeature()
    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(gx[i], gy[i])))
    f.setAttributes([gx[i], gy[i], float(est[i]), float(var[i])])
    features.append(f)

pr.addFeatures(features)
layer.updateExtents()

QgsProject.instance().addMapLayer(layer)

print("✓ DONE:", f"{DIENA:02d}.{MENESIS:02d}.  Model={MODEL}")
