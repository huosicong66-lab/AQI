import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from gplearn.genetic import SymbolicRegressor

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

print("Selected features for PLS+GEP:")
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

print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

scaler = StandardScaler()

X_train_val = np.vstack([X_train, X_val])
y_train_val = np.hstack([y_train, y_val])

scaler.fit(X_train_val)

X_train_val_scaled = scaler.transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

n_components = min(3, X_train_val_scaled.shape[1], X_train_val_scaled.shape[0])
pls = PLSRegression(n_components=n_components)

pls.fit(X_train_val_scaled, y_train_val)

Z_train_val = pls.transform(X_train_val_scaled)
Z_test = pls.transform(X_test_scaled)

print("PLS latent shapes:", Z_train_val.shape, Z_test.shape)

gep_model = SymbolicRegressor(
    population_size=500,
    generations=25,
    tournament_size=20,
    stopping_criteria=0.0,
    const_range=(-1.0, 1.0),
    init_depth=(2, 4),
    init_method='half and half',
    function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt', 'log'),
    metric='mse',
    parsimony_coefficient=1e-3,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=1.0,
    n_jobs=1,
    verbose=1,
    random_state=42
)

gep_model.fit(Z_train_val, y_train_val)

y_pred = gep_model.predict(Z_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

epsilon = 1e-6
denom = np.where(y_test == 0, epsilon, y_test)
mape = (np.abs((y_test - y_pred) / denom) * 100).mean()

r2 = r2_score(y_test, y_pred)

print("PLS + GEP Test Evaluation")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2  : {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "RMSE", "MAPE", "R2"],
    "Value": [mse, rmse, mape, r2]
})
metrics_df.to_excel("pls_gep_aqi_metrics.xlsx", index=False)
print("Saved: pls_gep_aqi_metrics.xlsx")

compare_df = pd.DataFrame({
    "Date_UTC": date_test,
    "AQI_True": y_test,
    "AQI_Pred": y_pred
})
compare_df.to_excel("pls_gep_aqi_actual_vs_pred.xlsx", index=False)
print("Saved: pls_gep_aqi_actual_vs_pred.xlsx")

plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, label="True AQI", linewidth=1)
plt.plot(range(len(y_test)), y_pred, label="PLS+GEP Pred", linestyle="--")
plt.title("PLS + GEP AQI Regression (Test Set)")
plt.xlabel("Test Sample Index (Time Order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()
