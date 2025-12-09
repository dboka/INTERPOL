import pandas as pd
import os

# ==== FAILA IESTATĪJUMI ====
input_file = r"C:\Users\deniss.boka\Downloads\Diennakts_videja_gaisa_temperatura_HOMOG(1).xlsx"
output_folder = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\CSV pa gadiem"

# Izveido mapi, ja tās nav
os.makedirs(output_folder, exist_ok=True)

# ==== Ielādē Excel ====
df = pd.read_excel(input_file)

# Pārliecināmies, ka kolonna "Gads" ir skaitlis
df["Gads"] = pd.to_numeric(df["Gads"], errors="coerce")

# ==== Filtrē 2000. gadu ====
df_2000 = df[df["Gads"] == 2024]

# ==== Saglabā CSV ====
output_csv = os.path.join(output_folder, "2024.csv")
df_2000.to_csv(output_csv, index=False, sep=";")

print("Gatavs! Saglabāts:", output_csv)