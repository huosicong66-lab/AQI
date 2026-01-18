import pandas as pd
import glob
import os


folder = "."

file_list = sorted(glob.glob(os.path.join(folder, "*.xls")))

dfs = []

for file in file_list:

    df = pd.read_excel(file)
    df["年份"] = os.path.basename(file).replace(".xls", "")  # 增加年份列（可选）
    dfs.append(df)


merged_df = pd.concat(dfs, ignore_index=True)


merged_df.to_excel("merged_2015_2024.xlsx", index=False)

print("merged_2015_2024.xlsx")
