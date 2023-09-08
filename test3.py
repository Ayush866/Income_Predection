import pandas as pd

df = pd.read_csv(r"adult.csv")
print(df['country'].unique().sort())