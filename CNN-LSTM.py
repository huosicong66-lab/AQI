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
df = pd.read_excel(file_path, dtype={"Date_UTC": str})

print("Columns:", df.columns.tolist())

time_col = "Date_UTC"
target_col = "AQI"

date_raw = df[time_col].astype(str).copy()

dt = pd.to_datetime(df[time_col], errors="coerce")
df["month"] = dt.dt.month
df["dayofyear"] = dt.dt.dayofyear

num_cols = [c for c in df.columns if c != time_col]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[num_cols] = (
    df[num_cols]
    .interpolate()
    .fillna(method="bfill")
    .fillna(method="ffill")
)

df = df.dropna(subset=[target_col]).copy()

all_cols = df.columns.tolist()

meteo_keywords = [
    "Temp", "TEMP", "Temperature", "Tair", "T_air",
    "RH", "Hum", "Humidity",
    "Wind", "WS", "WD", "Gust",
    "Press", "pressure", "Pressure", "Pres", "Baro",
    "Radi", "Solar", "SR"
]

def has_keyword(name, keywords):
    return any(k in name for k in keywords)

meteo_cols = []
for c in all_cols:
    if c in [time_col, target_col]:
        continue
    if has_keyword(c, meteo_keywords):
        meteo_cols.append(c)

meteo_cols = list(dict.fromkeys(meteo_cols))

for tcol in ["month", "dayofyear"]:
    if tcol not in meteo_cols and tcol in df.columns:
        meteo_cols.append(tcol)

print("Selected features:")
print(meteo_cols)

data = df[[target_col] + meteo_cols].dropna().copy()
dates_all = date_raw.loc[data.index].values

print("Data shape:", data.shape)

X_raw = data[meteo_cols].values.astype(float)
y_raw = data[target_col].values.astype(float)

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

X_seq = X_scaled.reshape(-1, 1, X_scaled.shape[1])

print("X_seq shape:", X_seq.shape)
print("y_scaled shape:", y_scaled.shape)

n = len(X_seq)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

X_train = X_seq[:n_train]
y_train_scaled = y_scaled[:n_train]

X_val   = X_seq[n_train:n_train + n_val]
y_val_scaled = y_scaled[n_train:n_train + n_val]

X_test  = X_seq[n_train + n_val:]
y_test_scaled = y_scaled[n_train + n_val:]
y_test_orig = y_raw[n_train + n_val:]
date_test = dates_all[n_train + n_val:]

print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

timesteps = X_train.shape[1]
n_features = X_train.shape[2]

inputs = Input(shape=(timesteps, n_features))
x = LSTM(32)(inputs)
x = Dropout(0.2)(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=["mape"]
)

model.summary()

es = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

rlr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=150,
    batch_size=32,
    callbacks=[es, rlr],
    verbose=1
)

y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()

mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)

epsilon = 1e-6
denom = np.where(y_test_orig == 0, epsilon, y_test_orig)
mape = (np.abs((y_test_orig - y_pred) / denom) * 100).mean()

r2 = r2_score(y_test_orig, y_pred)

print("Same-day AQI Evaluation (LSTM)")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2  : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE", "R2"],
    "Value": [mse, rmse, mape, r2]
})
metrics_df.to_excel("cnn_lstm_aqi_metrics.xlsx", index=False)
print("Saved: cnn_lstm_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test,
    "AQI_True": y_test_orig,
    "AQI_Pred": y_pred
})
compare_df.to_excel("cnn_lstm_aqi_actual_vs_pred.xlsx", index=False)
print("Saved: cnn_lstm_aqi_actual_vs_pred.xlsx")

plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test_orig)), y_test_orig, label="True AQI", linewidth=1)
plt.plot(range(len(y_test_orig)), y_pred, label="LSTM Pred", linestyle="--")
plt.title("Same-day AQI Prediction (Meteo-only LSTM)")
plt.xlabel("Test Sample Index (Time Order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()
