import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dense = Dense(1)

    def call(self, inputs):
        scores = self.dense(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

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
dates_all = date_raw.loc[df.index].values

all_cols = df.columns.tolist()

pollutant_keywords = [
    "PM2.5", "PM25", "PM_2_5", "PM10", "PM_10",
    "NO2", "NOx", "SO2", "O3", "CO"
]

meteo_keywords = [
    "Temp", "TEMP", "Temperature", "Tair", "T_air",
    "RH", "Hum", "Humidity",
    "Wind", "WS", "WD", "Gust",
    "Press", "pressure", "Pressure", "Pres", "Baro",
    "Radi", "Solar", "SR"
]

def has_kw(name, kws):
    return any(k in name for k in kws)

candidate_feats = []
for c in all_cols:
    if c in [time_col, target_col]:
        continue
    if has_kw(c, pollutant_keywords) or has_kw(c, meteo_keywords):
        candidate_feats.append(c)

candidate_feats = list(dict.fromkeys(candidate_feats))

for tcol in ["month", "dayofyear"]:
    if tcol in df.columns and tcol not in candidate_feats:
        candidate_feats.append(tcol)

print("Candidate features:")
print(candidate_feats)

data_all = df[[target_col] + candidate_feats].dropna().copy()

corr = data_all.corr()[target_col].drop(labels=[target_col])
K = min(12, len(corr))
top_feats = corr.abs().sort_values(ascending=False).head(K).index.tolist()

print("Selected features:")
for f in top_feats:
    print(f, corr[f])

use_cols = [target_col] + top_feats
data = data_all[use_cols].copy()
dates_all = date_raw.loc[data.index].values

print("Final data shape:", data.shape)

window_size = 30

values_all = data.values.astype(float)
aqi_all = data[target_col].values.astype(float)

scaler_X = MinMaxScaler()
values_all_scaled = scaler_X.fit_transform(values_all)

scaler_y = MinMaxScaler()
aqi_all_scaled = scaler_y.fit_transform(aqi_all.reshape(-1, 1)).flatten()

X_list, y_scaled_list, y_orig_list, date_target_list = [], [], [], []

for i in range(window_size, len(values_all_scaled)):
    X_list.append(values_all_scaled[i-window_size:i, :])
    y_scaled_list.append(aqi_all_scaled[i])
    y_orig_list.append(aqi_all[i])
    date_target_list.append(dates_all[i])

X_seq = np.array(X_list)
y_seq_scaled = np.array(y_scaled_list)
y_seq_orig = np.array(y_orig_list)
date_seq = np.array(date_target_list)

print("Sequence shapes:", X_seq.shape, y_seq_scaled.shape)

n = len(X_seq)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

X_train = X_seq[:n_train]
y_train_scaled = y_seq_scaled[:n_train]

X_val   = X_seq[n_train:n_train + n_val]
y_val_scaled = y_seq_scaled[n_train:n_train + n_val]

X_test  = X_seq[n_train + n_val:]
y_test_scaled = y_seq_scaled[n_train + n_val:]
y_test_orig   = y_seq_orig[n_train + n_val:]
date_test     = date_seq[n_train + n_val:]

print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

timesteps = X_train.shape[1]
n_features = X_train.shape[2]

inputs = Input(shape=(timesteps, n_features))

x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = Attention()(x)
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

print("Attention-LSTM Test Evaluation")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2  : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE", "R2"],
    "Value": [mse, rmse, mape, r2]
})
metrics_df.to_excel("att_lstm_aqi_metrics.xlsx", index=False)
print("Saved: att_lstm_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test,
    "AQI_True": y_test_orig,
    "AQI_Pred": y_pred
})
compare_df.to_excel("att_lstm_aqi_actual_vs_pred.xlsx", index=False)
print("Saved: att_lstm_aqi_actual_vs_pred.xlsx")

plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test_orig)), y_test_orig, label="True AQI", linewidth=1)
plt.plot(range(len(y_test_orig)), y_pred, label="Attention-LSTM Pred", linestyle="--")
plt.title("Attention-LSTM AQI Forecast (Test Set)")
plt.xlabel("Test Sample Index (Time Order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()
