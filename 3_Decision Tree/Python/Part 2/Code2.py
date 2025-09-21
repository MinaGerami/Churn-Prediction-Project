import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# خواندن داده‌ها و پیش‌پردازش
df = pd.read_csv("Data.csv")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("churn", axis=1)
y = df["churn"]
feature_names = X.columns.tolist()

# تقسیم داده‌ها به 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# آموزش مدل درخت تصمیم
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(X_train, y_train)

# تابع بازگشتی برای تولید متن ساده از درخت
def simple_tree_to_text(tree, feature_names, node_id=0, depth=0):
    indent = "  " * depth
    tree_ = tree.tree_

    if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
        name = feature_names[tree_.feature[node_id]]
        threshold = tree_.threshold[node_id]
        text = f"{indent}if {name} <= {threshold:.2f}:\n"
        text += simple_tree_to_text(tree, feature_names, tree_.children_left[node_id], depth + 1)
        text += f"{indent}else:  # if {name} > {threshold:.2f}\n"
        text += simple_tree_to_text(tree, feature_names, tree_.children_right[node_id], depth + 1)
        return text
    else:
        values = tree_.value[node_id][0]
        class_id = np.argmax(values)
        return f"{indent}return {class_id}\n"

# ذخیره متن ساده درخت در فایل txt
tree_text = simple_tree_to_text(tree, feature_names)
with open("simple_decision_tree.txt", "w") as f:
    f.write(tree_text)
