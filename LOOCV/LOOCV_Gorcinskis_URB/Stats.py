import pandas as pd

# ==== IESTATI CEĻU UZ CSV FAILU ====
csv_path = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\LOOCV_Gorcinskis_URB\LOOCV_summary_h10_URB.csv"

# ==== LATVIEŠU MĒNEŠU SECĪBA ====
month_order = [
    "Janvāris", "Februāris", "Marts", "Aprīlis", "Maijs", "Jūnijs",
    "Jūlijs", "Augusts", "Septembris", "Oktobris", "Novembris", "Decembris"
]

# ==== NOLASA CSV FAILU ====
df = pd.read_csv(csv_path)

# Pārbaude
print("Kolonnas failā:", df.columns.tolist())

# ==== SAKĀRTO PĒC MĒNEŠIEM ====
df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
df = df.sort_values("Month")

print("\n=== LOOCV dati pēc mēneša ===")
print(df)

# ==== APRĒĶINA VIDĒJOS RĀDĪTĀJUS ====
mean_mae = df["MAE"].mean()
mean_rmse = df["RMSE"].mean()
mean_bias = df["Bias"].mean()

summary = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "Bias"],
    "Mean": [mean_mae, mean_rmse, mean_bias]
})

print("\n=== KOPĒJĀS KĻŪDAS ===")
print(summary)

# ==== SAGLABĀ VIDĒJOS RĀDĪJUS ====
out_summary = csv_path.replace(".csv", "_summary.csv")
summary.to_csv(out_summary, index=False)

print("\nKopējie rādītāji saglabāti:", out_summary)

# ==== SAGLABĀ ARĪ SAKĀRTOTO CSV (ja vajag) ====
out_sorted = csv_path.replace(".csv", "_sorted.csv")
df.to_csv(out_sorted, index=False)
print("Sakārtotais fails saglabāts:", out_sorted)
