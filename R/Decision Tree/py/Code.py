from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("Data.csv")


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])


X = df.drop("churn", axis=1)
y = df["churn"]


tree = DecisionTreeClassifier(criterion='entropy')  # یا 'gini'
tree.fit(X, y)

importances = pd.Series(tree.feature_importances_, index=X.columns)
importances.sort_values(ascending=False)
