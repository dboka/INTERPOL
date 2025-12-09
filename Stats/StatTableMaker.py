import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# =====================================================
# LOAD LOOCV RESULTS
# =====================================================

df = pd.read_csv(
    r"C:\Users\deniss.boka\Interpol\data2024\LOOCV_daily_2024_Model2_FULL.csv",
    sep=";"
)

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month_name()
df["month_num"] = df["date"].dt.month

os.makedirs("loocv_plots", exist_ok=True)

# =====================================================
# TOTAL RMSE FOR FULL YEAR
# =====================================================

total_rmse = np.sqrt((df["error"]**2).mean())
total_mae = df["abs_error"].mean()
total_bias = df["error"].mean()

print("TOTAL RMSE:", total_rmse)
print("TOTAL MAE:", total_mae)
print("TOTAL BIAS:", total_bias)

# Save to file
with open("loocv_plots/total_stats.txt", "w") as f:
    f.write(f"Total RMSE: {total_rmse}\n")
    f.write(f"Total MAE: {total_mae}\n")
    f.write(f"Total Bias: {total_bias}\n")
# =====================================================
# ROLLING RMSE (30-day window)
# =====================================================

daily = df.groupby("date")["error"].apply(lambda x: np.sqrt((x**2).mean()))
rolling_rmse = daily.rolling(30, center=True).mean().reset_index()

fig = px.line(
    rolling_rmse,
    x="date",
    y="error",
    title="Rolling RMSE (30-day window)",
    markers=False
)

fig.write_html("loocv_plots/rmse_rolling_30d.html")
# =====================================================
# DAILY RMSE PLOT
# =====================================================

daily_rmse = df.groupby("date")["RMSE"].mean().reset_index()

fig = px.line(
    daily_rmse,
    x="date",
    y="RMSE",
    title="Daily RMSE Throughout 2023",
    markers=True
)
fig.write_html("loocv_plots/rmse_daily_2023.html")



# =====================================================
# DAILY MAE PLOT
# =====================================================

daily_mae = df.groupby("date")["MAE"].mean().reset_index()

fig = px.line(
    daily_mae,
    x="date",
    y="MAE",
    title="Daily MAE Throughout 2023",
    markers=True
)
fig.write_html("loocv_plots/mae_daily_2023.html")



# =====================================================
# DAILY BIAS PLOT
# =====================================================

daily_bias = df.groupby("date")["BIAS"].mean().reset_index()

fig = px.line(
    daily_bias,
    x="date",
    y="BIAS",
    title="Daily Bias (Model Over/Under Estimation) 2023",
    markers=True
)
fig.write_html("loocv_plots/bias_daily_2023.html")



# =====================================================
# BEST & WORST DAYS
# =====================================================

df_day_stats = df.groupby("date").agg(
    RMSE=("RMSE", "mean"),
    MAE=("MAE", "mean"),
    BIAS=("BIAS", "mean")
).reset_index()

best_days = df_day_stats.nsmallest(10, "RMSE")
worst_days = df_day_stats.nlargest(10, "RMSE")

best_days.to_csv("loocv_plots/best_days_2023.csv", index=False)
worst_days.to_csv("loocv_plots/worst_days_2023.csv", index=False)



# =====================================================
# RMSE PER MONTH
# =====================================================

monthly_rmse = df.groupby("month_num")["RMSE"].mean()

fig = px.bar(
    monthly_rmse,
    title="Mean RMSE per Month 2023",
    labels={"value": "RMSE", "month_num": "Month"}
)
fig.write_html("loocv_plots/monthly_rmse_2023.html")



# =====================================================
# GENERAL HISTOGRAM OF ERRORS
# =====================================================

fig = px.histogram(
    df,
    x="error",
    nbins=50,
    title="Distribution of Prediction Errors 2023"
)
fig.write_html("loocv_plots/error_histogram_2023.html")

fig = px.histogram(
    df,
    x="abs_error",
    nbins=50,
    title="Distribution of Absolute Errors 2023"
)
fig.write_html("loocv_plots/abs_error_histogram_2023.html")



# =====================================================
# BOXPLOT RMSE PER MONTH
# =====================================================

fig = px.box(
    df,
    x="month",
    y="abs_error",
    title="Absolute Error Distribution per Month 2023"
)
fig.update_xaxes(categoryorder="array", categoryarray=list(daily_rmse["date"].dt.month_name().unique()))
fig.write_html("loocv_plots/monthly_boxplot_2023.html")



# =====================================================
# STATION-LEVEL PERFORMANCE
# =====================================================

station_rmse = df.groupby("station")["error"].apply(lambda x: np.sqrt((x**2).mean())).reset_index()
station_rmse.columns = ["station", "RMSE"]

fig = px.bar(
    station_rmse.sort_values("RMSE"),
    x="station",
    y="RMSE",
    title="RMSE per Station (Full Year 2023)",
)
fig.write_html("loocv_plots/station_rmse_2023.html")



# =====================================================
# COMBINED DASHBOARD (Optional)
# =====================================================

dashboard = """
<html>
<head><title>LOOCV Dashboard 2023</title></head>
<body>
<h1>LOOCV Daily Model – Full Analysis</h1>

<ul>
<li><a href="rmse_daily_2023.html">Daily RMSE</a></li>
<li><a href="mae_daily_2023.html">Daily MAE</a></li>
<li><a href="bias_daily_2023.html">Daily Bias</a></li>
<li><a href="error_histogram_2023.html">Error Histogram</a></li>
<li><a href="abs_error_histogram_2023.html">Absolute Error Histogram</a></li>
<li><a href="monthly_rmse_2023.html">RMSE per Month</a></li>
<li><a href="monthly_boxplot_2023.html">Monthly Error Boxplot</a></li>
<li><a href="station_rmse_2023.html">RMSE by Station</a></li>
</ul>
<h2>Total Year Statistics</h2>
<pre>
Total RMSE, MAE, BIAS are stored in: total_stats_2023.txt
</pre>

<h2>Best 10 Days</h2>
<iframe src="best_days_2023.csv"></iframe>
<h2>Worst 10 Days</h2>
<iframe src="worst_days_2023.csv"></iframe>

</body>
</html>
"""

with open("loocv_plots/dashboard_2023.html", "w") as f:
    f.write(dashboard)

print("✓ ALL ANALYTICAL GRAPHS GENERATED IN loocv_plots/")
