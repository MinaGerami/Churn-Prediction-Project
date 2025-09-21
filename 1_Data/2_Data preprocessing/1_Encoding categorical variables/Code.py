import pandas as pd

df = pd.read_csv("Test.csv")

categorical_columns = ['state', 'area_code', 'international_plan', 'voice_mail_plan']

df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns, dtype=int)

df_encoded.to_csv("Processed_Test.csv", index=False)

