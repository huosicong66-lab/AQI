import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr

# ---------- 1. Load data ----------
data_path = "ALL.xlsx"
df = pd.read_excel(data_path)

# Target variable
y = df["AQI"]

# Features: remove AQI and date column
X = df.drop(columns=["AQI", "Date_UTC"], errors="ignore")

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# ---------- 2. Train Random Forest model ----------
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# ---------- 3. Compute SHAP values ----------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# ---------- 4. SHAP summary plot (beeswarm) ----------

plt.figure(figsize=(9, 7))
shap.summary_plot(
    shap_values,
    X,
    max_display=15,
    show=False
)

ax = plt.gca()
ax.set_xlabel("SHAP value", fontsize=12)

plt.tight_layout()
plt.savefig("shap_summary_all_features.png", dpi=1000, bbox_inches="tight")
plt.close()

# ---------- 5. Select Top15 features by mean |SHAP| ----------
# Compute mean absolute SHAP value for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

top10 = importance_df.head(15)
top10_features = top10["feature"].tolist()

print("Top10 features:")
print(top10_features)

top10.to_excel("shap_top10_feature_importance.xlsx", index=False)

# ---------- 6. SHAP dependence plots for Top10 features ----------
for feat in top10_features:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feat,
        shap_values,
        X,
        show=False
    )
    plt.tight_layout()
    fname = f"shap_dependence_{feat}.png"
    # Replace illegal characters in file names
    fname = fname.replace("/", "_").replace(" ", "_")
    plt.savefig(fname, dpi=800, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)

# ---------- 7. Numerical relationship between Top10 features and AQI ----------
rows = []
for feat in top10_features:
    x = df[feat]

    # Remove missing values (align indices)
    mask = x.notna() & y.notna()
    xv = x[mask]
    yv = y[mask]

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(xv, yv)

    # Spearman correlation
    spearman_r, spearman_p = spearmanr(xv, yv)

    rows.append({
        "feature": feat,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    })

rel_df = pd.DataFrame(rows)
rel_df.to_excel("top15_feature_AQI_relationship.xlsx", index=False)

print("top10_feature_AQI_relationship.xlsx")
print(rel_df)
