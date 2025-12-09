import numpy as np
import gstools as gs
from qgis.core import QgsProject, QgsField, QgsFeature, QgsVectorLayer, QgsPointXY
from PyQt5.QtCore import QVariant

# --------------------------
# SELECT MONTH
# --------------------------
month = "Janvāris"

# --------------------------
# LOAD LAYERS
# --------------------------
st_layer = QgsProject.instance().mapLayersByName("STACIJAS_LKS92")[0]
grid_layer = QgsProject.instance().mapLayersByName("GRID_LKS92")[0]

# --------------------------
# STATION COORDINATES & VALUES
# --------------------------
st_x = []
st_y = []
st_val = []

# EXT DRIFT TRAIN VARIABLES
d_elev = []
d_to_sea = []

for f in st_layer.getFeatures():
    st_x.append(f["x"])
    st_y.append(f["y"])
    st_val.append(f[month])
    
    d_elev.append(f["elevation"])
    d_to_sea.append(f["to_sea"])

st_x = np.array(st_x)
st_y = np.array(st_y)
st_val = np.array(st_val)

cond_pos = np.vstack([st_x, st_y])
drift_train = [np.array(d_elev), np.array(d_to_sea)]

# --------------------------
# VARIOGRAM MODEL
# --------------------------
bin_center, gamma = gs.vario_estimate_unstructured(cond_pos, st_val)
model = gs.Gaussian(dim=2, var=np.var(st_val), len_scale=39500, nugget=0)
model.fit_variogram(bin_center, gamma)

krige = gs.krige.ExtDrift(
    model,
    cond_pos=cond_pos,
    cond_val=st_val,
    exact=True,
    ext_drift=drift_train
)

# --------------------------
# OUTPUT LAYER
# --------------------------
output = QgsVectorLayer("Point?crs=EPSG:3059", f"Kriging_ED_{month}", "memory")
pr = output.dataProvider()

pr.addAttributes([
    QgsField("x", QVariant.Double),
    QgsField("y", QVariant.Double),
    QgsField("est", QVariant.Double),
    QgsField("var", QVariant.Double)
])
output.updateFields()

# --------------------------
# INTERPOLATE ON GRID
# --------------------------
features = []

for f in grid_layer.getFeatures():
    gx = f["x"]
    gy = f["y"]
    
    # GRID DRIFT VALUES
    g_elev = f["h5"]       # grid elevation substitute
    g_to_sea = f["to_sea"]
    
    est, var = krige(
        [gx, gy],
        ext_drift=[np.array([g_elev]), np.array([g_to_sea])]
    )
    
    new_f = QgsFeature()
    new_f.setGeometry(f.geometry())
    new_f.setAttributes([gx, gy, float(est), float(var)])
    features.append(new_f)

pr.addFeatures(features)
output.updateExtents()

QgsProject.instance().addMapLayer(output)

print("DONE EXT-DRIFT KRIGING:", month)