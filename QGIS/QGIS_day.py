import pandas as pd
import numpy as np
import gstools as gs
from pyproj import Transformer
from qgis.core import QgsProject, QgsField, QgsFeature, QgsVectorLayer, QgsGeometry, QgsPointXY
from PyQt5.QtCore import QVariant

# =========================================================
# INPUT SETTINGS — CHOOSE DAY
# =========================================================
MENESIS = 1   # <-- Month number (1–12)
DIENA = 23    # <-- Day number (1–31)

stations_csv = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\CSV pa gadiem\2000_long.csv"
grid_csv     = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\1x1_LV_grid_2024_xy2.csv"

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
# ADD lat84 / lon84 (LKS92 → WGS84)
# =========================================================
transformer = Transformer.from_crs(3059, 4326, always_xy=True)

df_day["lon84"], df_day["lat84"] = transformer.transform(df_day["x"].to_numpy(),
                                                         df_day["y"].to_numpy())

# =========================================================
# TRAIN POS & DRIFTS
# =========================================================
st_x = df_day["x"].to_numpy()
st_y = df_day["y"].to_numpy()
st_val = df_day["value"].to_numpy()

cond_pos = np.vstack([st_x, st_y])

# TRAIN DRIFTS (CONFIG4)
drift_train = [
    df_day["elevation"].to_numpy(),
    df_day["to_sea"].to_numpy(),
    df_day["cont_tas"].to_numpy(),
    df_day["lat84"].to_numpy()
]

# =========================================================
# LOAD GRID (FULL LATVIA)
# =========================================================
df_grid = pd.read_csv(grid_csv)

df_grid["lon84"], df_grid["lat84"] = transformer.transform(df_grid["x"].to_numpy(),
                                                           df_grid["y"].to_numpy())

print("✓ Loaded GRID POINTS:", len(df_grid))

gx = df_grid["x"].to_numpy()
gy = df_grid["y"].to_numpy()
grid_pos = np.vstack([gx, gy])

# TEST DRIFTS (CONFIG4)
drift_test = [
    df_grid["h10"].to_numpy(),     # DEM (h10)
    df_grid["to_sea"].to_numpy(),
    df_grid["cont_tas"].to_numpy(),
    df_grid["lat84"].to_numpy()
]

# =========================================================
# VARIOGRAM MODEL
# =========================================================
bin_center, gamma = gs.vario_estimate_unstructured(cond_pos, st_val)

model = gs.Gaussian(
    dim=2,
    var=np.var(st_val),
    len_scale=39500,
    nugget=0
)
model.fit_variogram(bin_center, gamma)

# =========================================================
# EXT-DRIFT KRIGING
# =========================================================
krige = gs.krige.ExtDrift(
    model,
    cond_pos=cond_pos,
    cond_val=st_val,
    exact=True,
    ext_drift=drift_train
)

est, var = krige(grid_pos, ext_drift=drift_test)

# =========================================================
# CREATE QGIS LAYER
# =========================================================
layer = QgsVectorLayer("Point?crs=EPSG:3059", f"Kriging_{MENESIS:02d}_{DIENA:02d}", "memory")
pr = layer.dataProvider()

pr.addAttributes([
    QgsField("x", QVariant.Double),
    QgsField("y", QVariant.Double),
    QgsField("est", QVariant.Double),
    QgsField("var", QVariant.Double)
])
layer.updateFields()

features = []
for i in range(len(gx)):
    f = QgsFeature()
    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(gx[i], gy[i])))
    f.setAttributes([gx[i], gy[i], float(est[i]), float(var[i])])
    features.append(f)

pr.addFeatures(features)
layer.updateExtents()

QgsProject.instance().addMapLayer(layer)

print("✓ DAILY KRIGING DONE FOR:", f"{DIENA:02d}.{MENESIS:02d}.")