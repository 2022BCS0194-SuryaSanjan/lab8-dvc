import pandas as pd

df = pd.read_csv("housing.csv")

df_partial = df.head(5000)

df_partial.to_csv("data/housing.csv", index=False)
