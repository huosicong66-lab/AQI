import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from vmdpy import VMD
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

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

signal = aqi_df["AQI"].values.astype(float)
N = len(signal)
print("Valid AQI samples:", N)

alpha = 2000
tau = 0.0
K = 4
DC = 0
init = 1
tol = 1e-7

u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
u = u.T

print("VMD modes shape:", u.shape)

window_size = 24

X_list, y_list, date_list = [], [], []
dates = aqi_df[time_col].values

for i in range(window_size, N):
    X_list.append(u[i - window_size:i, :])
    y_list.append(signal[i])
    date_list.append(dates[i])

X_seq = np.array(X_list)
y_seq = np.array(y_list)
date_seq = np.array(date_list)

print("Sequence shapes:", X_seq.shape, y_seq.shape)

n = len(X_seq)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

X_train = X_seq[:n_train]
y_train = y_seq[:n_train]

X_val = X_seq[n_train:n_train + n_val]
y_val = y_seq[n_train:n_train + n_val]

X_test = X_seq[n_train + n_val:]
y_test = y_seq[n_train + n_val:]
date_test = date_seq[n_train + n_val:]

print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

timesteps = X_train.shape[1]
n_nodes = X_train.shape[2]

class SimpleGAT(Layer):
    def __init__(self, out_dim, **kwargs):
        super(SimpleGAT, self).__init__(**kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        K = input_shape[-1]
        self.W = self.add_weight(
            shape=(K, self.out_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.a = self.add_weight(
            shape=(2 * self.out_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        K = tf.shape(inputs)[2]

        H = tf.matmul(inputs, self.W)

        H_expanded = tf.expand_dims(H, axis=2)
        H_repeat_i = tf.repeat(H_expanded, K, axis=2)
        H_repeat_j = tf.repeat(H_expanded, K, axis=2)

        H_concat = tf.concat([H_repeat_i, H_repeat_j], axis=-1)

        e = tf.nn.leaky_relu(
            tf.tensordot(H_concat, self.a, axes=[-1, 0])
        )

        alpha = tf.nn.softmax(e, axis=2)

        H_gat = tf.reduce_sum(alpha * H_repeat_i, axis=2)
        return H_gat

inputs = Input(shape=(timesteps, n_nodes))

gat_dim = 16
x = SimpleGAT(out_dim=gat_dim)(inputs)
x = Bidirectional(LSTM(64))(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()

es = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

y_pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

epsilon = 1e-6
denom = np.where(y_test == 0, epsilon, y_test)
mape = (np.abs((y_test - y_pred) / denom) * 100).mean()

print("VMD-GAT-BiLSTM Test Evaluation")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE"],
    "Value": [mse, rmse, mape]
})
metrics_df.to_excel("vmd_gat_bilstm_aqi_metrics.xlsx", index=False)
print("Saved: vmd_gat_bilstm_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test,
    "AQI_True": y_test,
    "AQI_Pred": y_pred
})
compare_df.to_excel(
    "vmd_gat_bilstm_aqi_actual_vs_pred.xlsx",
    index=False
)
print("Saved: vmd_gat_bilstm_aqi_actual_vs_pred.xlsx")

plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, label="True AQI", linewidth=1)
plt.plot(range(len(y_test)), y_pred, label="VMD-GAT-BiLSTM Pred", linestyle="--")
plt.title("VMD-GAT-BiLSTM AQI Forecast (Test Set)")
plt.xlabel("Test Sample Index (Time Order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()
