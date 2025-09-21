# Telecom Customer Churn Prediction

This project focuses on predicting customer churn in the telecommunications industry using machine learning techniques. The goal is to analyze customer behavior and identify patterns that lead to churn.

## Importance of Churn Prediction
Churn prediction helps telecom companies identify customers likely to leave, allowing for timely retention strategies. It reduces customer acquisition costs and boosts long-term profitability by focusing on customer loyalty and satisfaction.

## Completed Steps
1. Converted categorical variables into numerical format using encoding techniques.
2. Normalized all numerical features using Min-Max scaling to bring values into the [0, 1] range.
3. Split the dataset into training and testing sets, and implemented the K-Nearest Neighbors (KNN) algorithm for prediction. Results were saved and evaluated.
4. Applied Leave-One-Out Cross-Validation (LOOCV) with KNN to assess model performance on each individual sample.
5. Performed Leave-N-Out Cross-Validation with configurable test size to evaluate the model across multiple folds. Generated detailed metrics including TP, TN, FP, FN, accuracy, recall, precision, F1 score, and others.
Implemented Decision Tree models for classification:
    - **Python version:** Used `sklearn.tree.DecisionTreeClassifier` to train and evaluate a decision tree on the dataset. Generated metrics include accuracy, sensitivity (recall), specificity, precision, NPV, F1 score, and error rate.
    - **R version:** Used `rpart` and `rpart.plot` packages to build and visualize a decision tree. Evaluated the model using the same metrics to compare performance with Python and KNN results.

## Work in Progress
The project is currently in progress.
