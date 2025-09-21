import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("X.csv")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df.to_csv("Normalized_Train.csv", index=False)