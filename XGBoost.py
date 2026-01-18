import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

file_path = "ALL.xlsx"
df = pd.read_excel(file_path, dtype={"Date_UTC": str})

print("Columns:", df.columns.tolist())

time_col = "Date_UTC"
target_col = "AQI"

date_raw = df[time_col].astype(str).copy()

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

feature_cols = [
    "NO2_ugkg",
    "Ozone_ugkg",
    "Temperature_2m_Avg_C",
    "Relative_Humidity_pct"
]

feature_cols = [c for c in feature_cols if c in df.columns]

print("Selected features:")
print(feature_cols)

X_all = df[feature_cols].values
y_all = df[target_col].values
dates_all = date_raw.loc[df.index].values

print("Samples:", len(X_all), "Features:", X_all.shape[1])

n = len(X_all)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

X_train = X_all[:n_train]
y_train = y_all[:n_train]

X_val   = X_all[n_train:n_train + n_val]
y_val   = y_all[n_train:n_train + n_val]

X_test  = X_all[n_train + n_val:]
y_test  = y_all[n_train + n_val:]
date_test = dates_all[n_train + n_val:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scaler = StandardScaler()

X_train_val = np.vstack([X_train, X_val])
y_train_val = np.hstack([y_train, y_val])

scaler.fit(X_train_val)

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
X_train_val_scaled = scaler.transform(X_train_val)

model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric="rmse",
    verbose=False,
    early_stopping_rounds=80
)

print("Best iteration:", model.best_iteration)

best_n_estimators = model.best_iteration + 1

final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)

final_model.fit(X_train_val_scaled, y_train_val)

y_pred = final_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

epsilon = 1e-6
denom = np.where(y_test == 0, epsilon, y_test)
mape = (np.abs((y_test - y_pred) / denom) * 100).mean()

r2 = r2_score(y_test, y_pred)

print("===== XGBoost Test Evaluation =====")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2  : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE", "R2"],
    "Value": [mse, rmse, mape, r2]
})
metrics_df.to_excel("xgb_aqi_metrics.xlsx", index=False)
print("Saved: xgb_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test,
    "AQI_True": y_test,
    "AQI_Pred": y_pred
})
compare_df.to_excel("xgb_aqi_actual_vs_pred.xlsx", index=False)
print("Saved: xgb_aqi_actual_vs_pred.xlsx")

plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, label="True AQI", linewidth=1)
plt.plot(range(len(y_test)), y_pred, label="XGBoost Pred", linestyle="--")
plt.title("XGBoost AQI Regression (Test Set)")
plt.xlabel("Test Sample Index (Time Order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()
