import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
test_size = 1250 
# -----------------------------
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")
# -----------------------------
X_train = X.iloc[:-test_size, :]
X_test = X.iloc[-test_size:, :]

Y_train = Y.iloc[:-test_size, :]
Y_test = Y.iloc[-test_size:, :]
# -----------------------------
X_train.to_csv("TrainX.csv", index=False)
X_test.to_csv("TestX.csv", index=False)
Y_train.to_csv("TrainY.csv", index=False)
Y_test.to_csv("TestY.csv", index=False)
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train.values.ravel())  
# -----------------------------
Y_pred = knn.predict(X_test)
# -----------------------------
results = pd.DataFrame({
    'Actual': Y_test.values.ravel(),
    'Predicted': Y_pred
})

results.to_excel("Result.xlsx", index=False)

# -----------------------------
# -----------------------------

df_result = pd.read_excel("Result.xlsx")

df_result["Actual"] = df_result["Actual"].astype(str).str.lower()
df_result["Predicted"] = df_result["Predicted"].astype(str).str.lower()

TP = ((df_result["Actual"] == "yes") & (df_result["Predicted"] == "yes")).sum()
TN = ((df_result["Actual"] == "no") & (df_result["Predicted"] == "no")).sum()
FP = ((df_result["Actual"] == "no") & (df_result["Predicted"] == "yes")).sum()
FN = ((df_result["Actual"] == "yes") & (df_result["Predicted"] == "no")).sum()

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # TPR, Recall
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # TNR
ppv = TP / (TP + FP) if (TP + FP) != 0 else 0          # Precision
npv = TN / (TN + FN) if (TN + FN) != 0 else 0
f1_score = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) != 0 else 0
# -----------------------------
eval_data = {
    "Metric": ["TP", "TN", "FP", "FN", "Accuracy", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV", "F1 Score"],
    "Value": [TP, TN, FP, FN, accuracy, sensitivity, specificity, ppv, npv, f1_score]
}

df_eval = pd.DataFrame(eval_data)

df_eval.to_excel("Evaluation.xlsx", index=False)