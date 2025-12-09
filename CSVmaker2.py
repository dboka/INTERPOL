import pandas as pd
import os

# ===== Faili =====
df_daily_path = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\CSV pa gadiem\2024.csv"
df_meta_path  = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\staciju_dati_norma3.csv"
output_path   = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\CSV pa gadiem\2024_long.csv"

# ===== Ielasa dienu failu (wide) =====
df = pd.read_csv(df_daily_path, sep=";")

# Identifikācijas kolonnas
id_cols = ["Gads", "Menesis", "Diena"]

# Stacijas kolonnas
station_cols = [c for c in df.columns if c not in id_cols]

# ===== Wide -> Long =====
df_long = df.melt(
    id_vars=id_cols,
    value_vars=station_cols,
    var_name="gh_id",
    value_name="value"
)

# ===== Ielasa staciju meta datus =====
df_meta = pd.read_csv(df_meta_path, sep=";")

# Paturam vajadzīgās kolonnas
keep_cols = [
    "gh_id", "name", "lon", "lat", "elevation", "x", "y",
    "to_sea", "cont_tas", "h1", "h5", "h10"
]

df_meta = df_meta[keep_cols]

# ===== Apvienojam dienu datus ar meta =====
df_final = df_long.merge(df_meta, on="gh_id", how="left")

# Izmet NaN vērtības
df_final = df_final.dropna(subset=["value"])

# ===== Saglabā =====
df_final.to_csv(output_path, index=False, sep=";")

print("GATAVS! Saglabāts:", output_path)
