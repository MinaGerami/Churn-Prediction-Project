import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

# -----------------------------
Y = Y.astype(str).apply(lambda x: x.str.lower())

# -----------------------------
TP = TN = FP = FN = 0
k = 5

for i in range(len(X)):
    X_train = X.drop(i)
    Y_train = Y.drop(i)
    X_test = X.iloc[[i]]
    Y_true = Y.iloc[i].values[0]

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train.values.ravel())

    Y_pred = model.predict(X_test)[0]

    if Y_true == "yes" and Y_pred == "yes":
        TP += 1
    elif Y_true == "no" and Y_pred == "no":
        TN += 1
    elif Y_true == "yes" and Y_pred == "no":
        FN += 1
    elif Y_true == "no" and Y_pred == "yes":
        FP += 1

# -----------------------------
total = TP + TN + FP + FN
accuracy = (TP + TN) / total if total != 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
npv = TN / (TN + FN) if (TN + FN) != 0 else 0
f1_score = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) != 0 else 0
error_rate = (FP + FN) / total if total != 0 else 0
# -----------------------------
metrics = {
    "Metric": ["TP", "TN", "FP", "FN", "Accuracy", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV", "F1 Score", "Error Rate"],
    "Value": [TP, TN, FP, FN, accuracy, sensitivity, specificity, ppv, npv, f1_score, error_rate]
}

df_metrics = pd.DataFrame(metrics)
df_metrics.to_excel("Evaluation_LOOCV.xlsx", index=False)