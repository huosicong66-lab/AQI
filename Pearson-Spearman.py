import pandas as pd
from scipy.stats import pearsonr, spearmanr


path = "ALL.xlsx"
df = pd.read_excel(path)

y = df["AQI"]


X = df.drop(columns=["AQI", "Date_UTC"], errors="ignore")

rows = []

for feat in X.columns:
    x = df[feat]
    mask = x.notna() & y.notna()
    xv = x[mask]
    yv = y[mask]

    # Pearson
    pearson_r, pearson_p = pearsonr(xv, yv)
    # Spearman
    spearman_r, spearman_p = spearmanr(xv, yv)

    rows.append({
        "feature": feat,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    })


out_df = pd.DataFrame(rows)
out_df.to_excel("all_features_AQI_relationship.xlsx", index=False)

print("all_features_AQI_relationship.xlsx")
print(out_df)
