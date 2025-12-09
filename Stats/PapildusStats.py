
import os
import pandas as pd
import numpy as np

# ======================================
# Iestatījumi – norādi savu mapi
# ======================================
LOOCV_DIR = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\LOOCV_Gorcinskis_LAT"

# ======================================
# Funkcijas statistikai
# ======================================
def rmse(x):
    return np.sqrt(np.mean(x**2))

def mae(x):
    return np.mean(np.abs(x))

def bias(x):
    return np.mean(x)

# ======================================
# Nolasa visus CSV no mapes
# ======================================
all_files = [f for f in os.listdir(LOOCV_DIR) if f.endswith(".csv")]
df_list = []

for f in all_files:
    path = os.path.join(LOOCV_DIR, f)
    df = pd.read_csv(path)
    df["source_file"] = f   # lai zinātu, no kura mēneša
    df_list.append(df)

# Apvieno vienā DataFrame
all_df = pd.concat(df_list, ignore_index=True)

# ======================================
# Aprēķina papildus statistikas kolonnas
# ======================================
all_df["pct_error"] = 100 * all_df["error"] / all_df["true"]

# ======================================
# Statistika pa stacijām
# ======================================
station_stats = all_df.groupby("station").agg({
    "error": [rmse, mae, bias],
    "abs_error": ["max", "mean"],
    "pct_error": "mean"
})

station_stats.columns = [
    "RMSE",
    "MAE",
    "Bias",
    "Max_Abs_Error",
    "Mean_Abs_Error",
    "Pct_Error_Mean"
]

# ======================================
# Statistika pa mēnešiem
# ======================================
month_stats = all_df.groupby("month").agg({
    "error": [rmse, mae, bias],
    "abs_error": ["max", "mean"],
    "pct_error": "mean"
})

month_stats.columns = [
    "RMSE",
    "MAE",
    "Bias",
    "Max_Abs_Error",
    "Mean_Abs_Error",
    "Pct_Error_Mean"
]

# ======================================
# Saglabā failus
# ======================================
station_stats.to_csv(os.path.join(LOOCV_DIR, "LOOCV_station_stats.csv"), sep=";")
month_stats.to_csv(os.path.join(LOOCV_DIR, "LOOCV_month_stats.csv"), sep=";")

print("DONE!")
print("Saglabāti:")
print(" - LOOCV_station_stats.csv")
print(" - LOOCV_month_stats.csv")
