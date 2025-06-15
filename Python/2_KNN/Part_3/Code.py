import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# --------------------------
N = 10
k = 5

X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv").astype(str).apply(lambda x: x.str.lower())

X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)
# --------------------------
TN_list, TP_list, FN_list, FP_list = [], [], [], []
# --------------------------
for i in range(0, len(X), N):
    X_test = X.iloc[i:i+N]
    Y_test = Y.iloc[i:i+N].values.ravel()

    X_train = X.drop(index=range(i, min(i+N, len(X))))
    Y_train = Y.drop(index=range(i, min(i+N, len(Y)))).values.ravel()


    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)


    TP = TN = FP = FN = 0
    for y_true, y_pred in zip(Y_test, Y_pred):
        if y_true == "yes" and y_pred == "yes":
            TP += 1
        elif y_true == "no" and y_pred == "no":
            TN += 1
        elif y_true == "yes" and y_pred == "no":
            FN += 1
        elif y_true == "no" and y_pred == "yes":
            FP += 1


    TP_list.append(TP)
    TN_list.append(TN)
    FN_list.append(FN)
    FP_list.append(FP)
# --------------------------
df_TN = pd.DataFrame({"TN": TN_list})
df_TP = pd.DataFrame({"TP": TP_list})
df_FN = pd.DataFrame({"FN": FN_list})
df_FP = pd.DataFrame({"FP": FP_list})


with pd.ExcelWriter("CrossVal_Details.xlsx") as writer:
    df_TN.to_excel(writer, sheet_name="TN", index=False)
    df_TP.to_excel(writer, sheet_name="TP", index=False)
    df_FN.to_excel(writer, sheet_name="FN", index=False)
    df_FP.to_excel(writer, sheet_name="FP", index=False)
# --------------------------
TP_sum, TN_sum, FP_sum, FN_sum = sum(TP_list), sum(TN_list), sum(FP_list), sum(FN_list)
total = TP_sum + TN_sum + FP_sum + FN_sum

accuracy = (TP_sum + TN_sum) / total if total != 0 else 0
sensitivity = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) != 0 else 0
specificity = TN_sum / (TN_sum + FP_sum) if (TN_sum + FP_sum) != 0 else 0
ppv = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) != 0 else 0
npv = TN_sum / (TN_sum + FN_sum) if (TN_sum + FN_sum) != 0 else 0
f1_score = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) != 0 else 0
error_rate = (FP_sum + FN_sum) / total if total != 0 else 0

summary = {
    "Metric": ["TP", "TN", "FP", "FN", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1 Score", "Error Rate"],
    "Value": [TP_sum, TN_sum, FP_sum, FN_sum, accuracy, sensitivity, specificity, ppv, npv, f1_score, error_rate]
}
df_summary = pd.DataFrame(summary)

df_summary.to_excel("CrossVal_Summary.xlsx", index=False)

