import pandas as pd

df = pd.read_csv("Train.csv")

state_dummies = pd.get_dummies(df['state'], prefix='state', dtype=int)

df.drop('state', axis=1, inplace=True)

df = pd.concat([df, state_dummies], axis=1)

df.to_csv("Processed_State_Train.csv", index=False)