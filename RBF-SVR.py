import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

file_path = "ALL.xlsx"
df = pd.read_excel(file_path, dtype={"Date_UTC": str})

print("Columns:", df.columns.tolist())

time_col = "Date_UTC"
target_col = "AQI"

date_raw = df[time_col].astype(str).copy()

aqi = pd.to_numeric(df[target_col], errors="coerce")
aqi = aqi.interpolate()
aqi = aqi.dropna()

aqi_df = pd.DataFrame({"AQI": aqi})
aqi_df[time_col] = date_raw.loc[aqi_df.index].values

values = aqi_df["AQI"].values.astype(float)
dates = aqi_df[time_col].values

print("Valid AQI samples:", len(values))

all_cols = df.columns.tolist()
feature_cols = [c for c in all_cols if c not in [time_col, target_col]]

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

meteo_cols = [
    c for c in feature_cols
    if (not is_pollutant(c)) and is_meteo(c)
]

if len(meteo_cols) == 0:
    print("No meteorological features detected.")
else:
    print("Meteorological features:", meteo_cols)

for c in meteo_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if len(meteo_cols) > 0:
    df[meteo_cols] = (
        df[meteo_cols]
        .interpolate()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

meteo_df = df.loc[aqi_df.index, meteo_cols] if len(meteo_cols) > 0 else None

window_size = 48

X_list, y_list, date_list = [], [], []

for i in range(window_size, len(values)):
    lag_feats = values[i - window_size:i]

    if meteo_df is not None:
        met_current = meteo_df.iloc[i].values
        feats = np.concatenate([lag_feats, met_current])
    else:
        feats = lag_feats

    X_list.append(feats)
    y_list.append(values[i])
    date_list.append(dates[i])

X = np.array(X_list)
y = np.array(y_list)
date_seq = np.array(date_list)

print("Feature shape:", X.shape, y.shape)

n = len(X)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

X_train = X[:n_train]
y_train = y[:n_train]

X_val   = X[n_train:n_train + n_val]
y_val   = y[n_train:n_train + n_val]

X_test  = X[n_train + n_val:]
y_test  = y[n_train + n_val:]
date_test = date_seq[n_train + n_val:]

print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

scaler_X = StandardScaler()
scaler_X.fit(X_train)

X_train_scaled = scaler_X.transform(X_train)
X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
scaler_y.fit(y_train.reshape(-1, 1))

y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

C_list = [1, 10, 100, 1000]
gamma_list = [1e-3, 1e-2, 1e-1, 1]
eps_list = [0.01, 0.05, 0.1]

best_mse_val = np.inf
best_cfg = None

print("Grid search for SVR hyperparameters")
for C in C_list:
    for gamma in gamma_list:
        for eps in eps_list:
            svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps)
            svr.fit(X_train_scaled, y_train_scaled)
            y_val_pred_scaled = svr.predict(X_val_scaled)
            y_val_pred = scaler_y.inverse_transform(
                y_val_pred_scaled.reshape(-1, 1)
            ).ravel()
            mse_val = mean_squared_error(y_val, y_val_pred)

            print(f"C={C}, gamma={gamma}, eps={eps} -> Val MSE={mse_val:.4f}")

            if mse_val < best_mse_val:
                best_mse_val = mse_val
                best_cfg = (C, gamma, eps)

print(f"Best config: C={best_cfg[0]}, gamma={best_cfg[1]}, epsilon={best_cfg[2]}, Val MSE={best_mse_val:.4f}")

C_best, gamma_best, eps_best = best_cfg

X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
y_trainval_scaled = np.hstack([y_train_scaled, y_val_scaled])

svr_best = SVR(
    kernel="rbf",
    C=C_best,
    gamma=gamma_best,
    epsilon=eps_best
)
svr_best.fit(X_trainval_scaled, y_trainval_scaled)

y_pred_test_scaled = svr_best.predict(X_test_scaled)
y_pred_test = scaler_y.inverse_transform(
    y_pred_test_scaled.reshape(-1, 1)
).ravel()

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

epsilon = 1e-6
denom = np.where(y_test == 0, epsilon, y_test)
mape = (np.abs((y_test - y_pred_test) / denom) * 100).mean()

r2 = r2_score(y_test, y_pred_test)

print("SVR Test Evaluation")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2  : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE", "R2"],
    "Value": [mse, rmse, mape, r2]
})
metrics_df.to_excel("rbf_svr_aqi_metrics.xlsx", index=False)
print("Saved: rbf_svr_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test,
    "AQI_True": y_test,
    "AQI_Pred": y_pred_test
})
compare_df.to_excel("rbf_svr_aqi_actual_vs_pred.xlsx", index=False)
print("Saved: rbf_svr_aqi_actual_vs_pred.xlsx")

plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, label="True AQI", linewidth=1)
plt.plot(range(len(y_test)), y_pred_test, label="SVR Pred", linestyle="--")
plt.title("SVR AQI Forecasting (Test Set)")
plt.xlabel("Test Sample Index (Time Order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()
