import pandas as pd

df = pd.read_parquet(r"C:\Users\limm1\Downloads\train-00000-of-00001.parquet")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())