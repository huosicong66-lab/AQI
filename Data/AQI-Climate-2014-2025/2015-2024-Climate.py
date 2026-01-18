import pandas as pd
import glob
import os


folder = "."


file_list = [f"{folder}/{year}.xls" for year in range(2015, 2025)]

dfs = []

for file in file_list:
    if os.path.exists(file):

        df = pd.read_excel(file)
        df["年份"] = os.path.basename(file).replace(".xls", "")
        dfs.append(df)
    else:
        print("NO", file)

# 合并所有年份
merged_df = pd.concat(dfs, ignore_index=True)

# 保存输出
output_name = "merged_2015_2024.xlsx"
merged_df.to_excel(output_name, index=False)

print(output_name)
