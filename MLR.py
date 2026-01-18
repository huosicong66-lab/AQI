import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

file_path = "ALL.xlsx"
df = pd.read_excel(file_path, dtype={"Date_UTC": str})

print("Columns:")
print(df.columns)

time_col = "Date_UTC"
target_col = "AQI"

df = df.sort_values(by=time_col)

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df[target_col] = df[target_col].interpolate()
df[target_col] = df[target_col].clip(lower=0)

all_feature_cols = [c for c in df.columns if c not in [time_col, target_col]]

for c in all_feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[all_feature_cols] = (
    df[all_feature_cols]
    .interpolate()
    .fillna(method="bfill")
    .fillna(method="ffill")
)

pollutant_keywords = [
    "PM2.5", "PM10", "PM_2_5", "PM_10", "PM25", "PM_25", "PM",
    "NO2", "NOx", "NO", "SO2", "O3", "CO", "AQI", "Index", "Pollut"
]

def is_pollutant(col_name):
    return any(pk in col_name for pk in pollutant_keywords)

met_keywords = [
    "Wind", "wind", "WS", "WD", "Gust",
    "Press", "pressure", "Pressure", "Pres", "Baro",
    "Temp", "TEMP", "Temperature", "Tair", "T_air",
    "RH", "Hum", "Humidity", "Dew", "dew",
    "Radiation", "Radi", "Solar", "SR", "Global", "Net", "SW",
    "Precip", "Rain", "rain",
    "Cloud", "cloud",
    "Vis", "Visibility"
]

def is_meteo(col_name):
    return any(mk in col_name for mk in met_keywords)

candidate_meteo_cols = [
    c for c in all_feature_cols
    if (not is_pollutant(c)) and is_meteo(c)
]

if len(candidate_meteo_cols) < 3:
    non_pollutant_cols = [c for c in all_feature_cols if not is_pollutant(c)]
    candidate_meteo_cols = non_pollutant_cols[:min(8, len(non_pollutant_cols))]

print("Candidate meteorological features:")
print(candidate_meteo_cols)

df_clean = df[[time_col, target_col] + candidate_meteo_cols].dropna()

print("Cleaned samples:", len(df_clean))

dates = df_clean[time_col].values
y = df_clean[target_col].values.astype(float)
X_full = df_clean[candidate_meteo_cols].values.astype(float)

n = len(df_clean)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

X_train = X_full[:n_train]
y_train = y[:n_train]

X_val   = X_full[n_train:n_train + n_val]
y_val   = y[n_train:n_train + n_val]

X_test  = X_full[n_train + n_val:]
y_test  = y[n_train + n_val:]
date_test = dates[n_train + n_val:]

print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

train_df_for_corr = pd.DataFrame(X_train, columns=candidate_meteo_cols)
train_df_for_corr[target_col] = y_train

corrs = train_df_for_corr.corr()[target_col].drop(index=target_col)
corr_abs = corrs.abs().sort_values(ascending=False)

TOP_K = 40
selected_cols = list(corr_abs.index[:TOP_K])

print("Selected features:")
print(selected_cols)

X_train = train_df_for_corr[selected_cols].values
X_val   = pd.DataFrame(X_val, columns=candidate_meteo_cols)[selected_cols].values
X_test  = pd.DataFrame(X_test, columns=candidate_meteo_cols)[selected_cols].values

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

def calc_mape(y_true, y_pred, eps=1e-6):
    denom = np.where(y_true == 0, eps, y_true)
    return (np.abs((y_true - y_pred) / denom) * 100).mean()

alpha_list = [0.01, 0.1, 1.0, 10.0, 100.0]
best_alpha = None
best_val_mape = np.inf

for alpha in alpha_list:
    model_tmp = Ridge(alpha=alpha)
    model_tmp.fit(X_train_scaled, y_train)
    y_val_pred = model_tmp.predict(X_val_scaled)
    val_mape = calc_mape(y_val, y_val_pred)
    print(f"alpha={alpha} -> Val MAPE={val_mape:.4f}%")
    if val_mape < best_val_mape:
        best_val_mape = val_mape
        best_alpha = alpha

print(f"Best alpha={best_alpha}, Val MAPE={best_val_mape:.4f}%")

X_train_val = np.vstack([X_train, X_val])
y_train_val = np.hstack([y_train, y_val])

scaler_final = StandardScaler()
scaler_final.fit(X_train_val)

X_train_val_scaled = scaler_final.transform(X_train_val)
X_test_scaled_final = scaler_final.transform(X_test)

mlr = Ridge(alpha=best_alpha)
mlr.fit(X_train_val_scaled, y_train_val)

print("Final Ridge model fitted")
print("Number of features:", X_train_val_scaled.shape[1])

y_pred = mlr.predict(X_test_scaled_final)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = calc_mape(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Evaluation (Ridge, meteo-only)")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"R2   : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE", "R2"],
    "Value": [mse, rmse, mape, r2]
})
metrics_df.to_excel("mlr_meteo_aqi_metrics.xlsx", index=False)
print("Saved: mlr_meteo_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test,
    "AQI_True": y_test,
    "AQI_Pred": y_pred
})
compare_df.to_excel("mlr_aqi_actual_vs_pred.xlsx", index=False)
print("Saved: mlr_aqi_actual_vs_pred.xlsx")
