import numpy as np
import pandas as pd
import gstools as gs
from pyproj import Transformer
from qgis.core import QgsProject, QgsField, QgsFeature, QgsVectorLayer, QgsGeometry, QgsPointXY
from PyQt5.QtCore import QVariant

# =========================================================
# CHOOSE MONTH (COLUMN NAME IN STATION LAYER)
# =========================================================
MONTH = "Janvāris"   # <-- change here (must match station field)

# =========================================================
# LOAD QGIS LAYERS
# =========================================================
st_layer = QgsProject.instance().mapLayersByName("STACIJAS_LKS92")[0]
grid_layer = QgsProject.instance().mapLayersByName("GRID_LKS92")[0]

print("✓ Loaded Stations:", st_layer.featureCount())
print("✓ Loaded Grid:", grid_layer.featureCount())

# =========================================================
# PREPARE LAT84 (LKS92 → WGS84)
# =========================================================
transformer = Transformer.from_crs(3059, 4326, always_xy=True)

# =========================================================
# EXTRACT STATION DATA
# =========================================================
st_x = []
st_y = []
st_val = []

dr_elev = []
dr_to_sea = []
dr_cont = []
dr_lat = []

for f in st_layer.getFeatures():
    x = f["x"]
    y = f["y"]
    val = f[MONTH]

    st_x.append(x)
    st_y.append(y)
    st_val.append(val)

    dr_elev.append(f["elevation"])
    dr_to_sea.append(f["to_sea"])
    dr_cont.append(f["cont_tas"])

    lon, lat = transformer.transform(x, y)
    dr_lat.append(lat)

st_x = np.array(st_x)
st_y = np.array(st_y)
st_val = np.array(st_val)

cond_pos = np.vstack([st_x, st_y])

drift_train = [
    np.array(dr_elev),
    np.array(dr_to_sea),
    np.array(dr_cont),
    np.array(dr_lat)
]

print("✓ Train Drift Shapes:", [d.shape for d in drift_train])

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

krige = gs.krige.ExtDrift(
    model,
    cond_pos=cond_pos,
    cond_val=st_val,
    exact=True,
    ext_drift=drift_train
)

# =========================================================
# CREATE OUTPUT LAYER
# =========================================================
layer = QgsVectorLayer("Point?crs=EPSG:3059", f"Kriging_CFG4_{MONTH}", "memory")
pr = layer.dataProvider()

pr.addAttributes([
    QgsField("x", QVariant.Double),
    QgsField("y", QVariant.Double),
    QgsField("est", QVariant.Double),
    QgsField("var", QVariant.Double)
])
layer.updateFields()

# =========================================================
# INTERPOLATE GRID WITH 4 DRIFTS
# =========================================================
features = []

for f in grid_layer.getFeatures():
    gx = f["x"]
    gy = f["y"]

    # GRID DRIFTS
    g_h10 = f["h10"]
    g_to_sea = f["to_sea"]
    g_cont = f["cont_tas"]

    lon, lat = transformer.transform(gx, gy)
    g_lat84 = lat

    drift_test = [
        np.array([g_h10]),
        np.array([g_to_sea]),
        np.array([g_cont]),
        np.array([g_lat84])
    ]

    est, var = krige([gx, gy], ext_drift=drift_test)

    new_f = QgsFeature()
    new_f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(gx, gy)))
    new_f.setAttributes([gx, gy, float(est), float(var)])
    features.append(new_f)

pr.addFeatures(features)
layer.updateExtents()
QgsProject.instance().addMapLayer(layer)

print("✓ MONTHLY CONFIG4 KRIGING DONE FOR:", MONTH)
