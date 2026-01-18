import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

file_path = "ALL.xlsx"
time_col = "Date_UTC"
target_col = "AQI"

df = pd.read_excel(file_path, dtype={time_col: str})
print("Columns:", df.columns.tolist())

date_str = df[time_col].astype(str).copy()

dt = pd.to_datetime(df[time_col], errors="coerce")
df = df.loc[~dt.isna()].copy()
dt = dt.loc[df.index]
df["dt"] = dt

df = df.sort_values("dt")
date_str = date_str.loc[df.index]

for c in df.columns:
    if c not in [time_col, "dt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

num_cols = [c for c in df.columns if c not in [time_col, "dt"]]
df[num_cols] = df[num_cols].interpolate().bfill().ffill()

df = df.dropna(subset=[target_col]).copy()
date_str = date_str.loc[df.index]
print("Valid samples:", len(df))

pollutant_keywords = ["PM2.5", "PM25", "PM_2_5", "PM10", "PM_10",
                      "NO2", "NOx", "SO2", "O3", "CO"]
meteo_keywords = ["Temp", "TEMP", "Temperature", "Tair", "T_air",
                  "RH", "Hum", "Humidity",
                  "Wind", "WS", "WD", "Gust",
                  "Press", "pressure", "Pressure", "Pres", "Baro",
                  "Radi", "Solar", "SR"]

def has_kw(name, kws):
    name_low = name.lower()
    return any(k.lower() in name_low for k in kws)

poll_cols = []
meteo_cols = []

for c in df.columns:
    if c in [time_col, "dt", target_col]:
        continue
    if has_kw(c, pollutant_keywords):
        poll_cols.append(c)
    if has_kw(c, meteo_keywords):
        meteo_cols.append(c)

poll_cols = list(dict.fromkeys(poll_cols))
meteo_cols = list(dict.fromkeys(meteo_cols))

print("Pollutant features:", poll_cols)
print("Meteorological features:", meteo_cols)

if len(poll_cols) == 0 and len(meteo_cols) == 0:
    backup_feats = [c for c in num_cols if c != target_col]
    print("Fallback features:", backup_feats)
    poll_cols = backup_feats
    meteo_cols = []

lag_days = 7

data = pd.DataFrame(index=df.index)
data[target_col] = df[target_col]

for l in range(1, lag_days + 1):
    data[f"{target_col}_lag{l}"] = df[target_col].shift(l)
    for c in poll_cols:
        data[f"{c}_lag{l}"] = df[c].shift(l)

data["AQI_ma3"] = df[target_col].shift(1).rolling(3).mean()
data["AQI_ma7"] = df[target_col].shift(1).rolling(7).mean()

for c in poll_cols[:3]:
    data[f"{c}_ma3"] = df[c].shift(1).rolling(3).mean()
    data[f"{c}_ma7"] = df[c].shift(1).rolling(7).mean()

data["month"] = df["dt"].dt.month
data["dayofweek"] = df["dt"].dt.dayofweek
data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
data["dayofyear"] = df["dt"].dt.dayofyear
data["sin_doy"] = np.sin(2 * np.pi * data["dayofyear"] / 365.0)
data["cos_doy"] = np.cos(2 * np.pi * data["dayofyear"] / 365.0)

for c in meteo_cols:
    data[f"{c}_lead1"] = df[c].shift(-1)

data = data.dropna().copy()
date_supervised = date_str.loc[data.index]

print("Feature matrix shape:", data.shape)

feature_cols = [c for c in data.columns if c != target_col]
X_all = data[feature_cols].values
y_all = data[target_col].values

n = len(X_all)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

X_train_raw = X_all[:n_train]
y_train = y_all[:n_train]

X_val_raw = X_all[n_train:n_train + n_val]
y_val = y_all[n_train:n_train + n_val]

X_test_raw = X_all[n_train + n_val:]
y_test = y_all[n_train + n_val:]
date_test = date_supervised.iloc[n_train + n_val:]

print(f"Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

scaler_X = MinMaxScaler()
scaler_X.fit(X_train_raw)

X_train_scaled = scaler_X.transform(X_train_raw)
X_val_scaled = scaler_X.transform(X_val_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

scaler_y = MinMaxScaler()
scaler_y.fit(y_train.reshape(-1, 1))

y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

n_features = X_train_scaled.shape[1]

X_train = X_train_scaled.reshape(-1, 1, n_features)
X_val = X_val_scaled.reshape(-1, 1, n_features)
X_test = X_test_scaled.reshape(-1, 1, n_features)

print("X shapes:", X_train.shape, X_val.shape, X_test.shape)

inputs = Input(shape=(1, n_features))

x = LSTM(64, return_sequences=False)(inputs)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(16, activation="relu")(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()

es = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

rlr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=200,
    batch_size=32,
    callbacks=[es, rlr],
    verbose=1
)

y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

epsilon = 1e-6
denom = np.where(y_test == 0, epsilon, y_test)
mape = (np.abs((y_test - y_pred) / denom) * 100).mean()

r2 = r2_score(y_test, y_pred)

print("LSTM Next-day AQI Evaluation")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2  : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE", "R2"],
    "Value": [mse, rmse, mape, r2]
})
metrics_df.to_excel("lstm_tabular_nextday_aqi_metrics.xlsx", index=False)
print("Saved: lstm_tabular_nextday_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test.values,
    "AQI_True": y_test,
    "AQI_Pred": y_pred
})
compare_df.to_excel("att_lstm_aqi_actual_vs_pred.xlsx", index=False)
print("Saved: att_lstm_aqi_actual_vs_pred.xlsx")

plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, label="True AQI", linewidth=1)
plt.plot(range(len(y_test)), y_pred, label="LSTM Next-day Pred", linestyle="--")
plt.title("LSTM Next-day AQI Forecast (Tabular Features)")
plt.xlabel("Test Sample Index (Time Order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()
