from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd

# خواندن و پیش‌پردازش
df = pd.read_csv("Data.csv")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("churn", axis=1)
y = df["churn"]

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# آموزش مدل
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(X_train, y_train)

# پیش‌بینی روی داده تست
y_pred = tree.predict(X_test)

# ذخیره در فایل اکسل
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_excel("prediction_results.xlsx", index=False)

# محاسبه TP, TN, FP, FN
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# نمایش نتایج
print("True Positives (TP):", tp)
print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)








from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.figure(figsize=(150, 50))
plot_tree(tree,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10)

plt.savefig("decision_tree.png", dpi=300)  # ذخیره به فایل









import matplotlib
matplotlib.use('Agg')
import numpy as np

# تعداد گره‌ها و آرایه‌ی feature هر گره (درخت)
tree_features = tree.tree_.feature  # -2 برای برگ‌ها، >=0 برای شاخص ویژگی

# شمارش تعداد تکرار هر ویژگی (ignore -2)
feature_counts = {}
for f in tree_features:
    if f >= 0:
        feature_counts[f] = feature_counts.get(f, 0) + 1

# نام ویژگی‌ها
feature_names = list(X.columns)

# ساخت DataFrame با نام، تعداد تکرار، امتیاز اهمیت
data = []
total_nodes = tree.tree_.node_count
for i, name in enumerate(feature_names):
    count = feature_counts.get(i, 0)
    importance = tree.feature_importances_[i]
    ratio = count / total_nodes
    data.append({
        'Feature': name,
        'Count_in_Tree': count,
        'Importance': importance,
        'Ratio_in_Tree': ratio,
        'Notes': ''  # اینجا می‌تونی توضیح یا هر متنی اضافه کنی
    })

importance_detail_df = pd.DataFrame(data).sort_values(by='Importance', ascending=False)

# ذخیره در اکسل
importance_detail_df.to_excel('feature_importance_detail.xlsx', index=False)

print(importance_detail_df)




import numpy as np
import pandas as pd

# شمارش تکرار هر ویژگی در گره‌های درخت
tree_features = tree.tree_.feature  # -2 یعنی برگ

feature_counts = {}
for f in tree_features:
    if f >= 0:
        feature_counts[f] = feature_counts.get(f, 0) + 1

feature_names = list(X.columns)
total_nodes = tree.tree_.node_count

data = []
for i, name in enumerate(feature_names):
    count = feature_counts.get(i, 0)
    importance = tree.feature_importances_[i]
    ratio = count / total_nodes
    data.append({
        'Feature': name,
        'Count_in_Tree': count,
        'Importance': importance,
        'Ratio_in_Tree': ratio,
        'Notes': ''
    })

importance_detail_df = pd.DataFrame(data).sort_values(by='Importance', ascending=False)

# ذخیره در فایل اکسل بدون چاپ
importance_detail_df.to_excel('feature_importance_detail.xlsx', index=False)























from sklearn.tree import export_text

# تولید کد شبه if/else
tree_rules = export_text(tree, feature_names=list(X.columns))

print("def predict_tree(input):")
for line in tree_rules.split('\n'):
    indent_level = line.count('|   ')
    content = line.replace('|   ', '').strip()

    if 'class:' in content:
        class_value = content.split('class: ')[1]
        print('    ' * (indent_level + 1) + f"return '{class_value}'")
    elif '<=' in content:
        feature, threshold = content.split(' <= ')
        print('    ' * (indent_level + 1) + f"if input['{feature}'] <= {threshold}:")
    elif '>' in content:  # فقط اگر حالت بالا نبود
        feature, threshold = content.split(' > ')
        print('    ' * (indent_level + 1) + f"else:  # input['{feature}'] > {threshold}")
