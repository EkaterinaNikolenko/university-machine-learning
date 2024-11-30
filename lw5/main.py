import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/bank-additional-full.csv', sep=';')

print(f'First five rows of the dataframe:\n {df.head()}')
print(f'Summary of the dataframe:\n {df.info()}')
print(f'Shape of data:\n {df.shape}')

# Replace 'unknown' with np.nan for specified columns
df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']] = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']].replace('unknown', np.nan)

# Replace 'nonexistent' with np.nan for the 'poutcome' column
df[['poutcome']] = df[['poutcome']].replace('nonexistent', np.nan)

# Replace 999 with np.nan for the 'pdays' column
df[['pdays']] = df[['pdays']].replace(999, np.nan)

print(f'Number of missing values in each column:\n {df.isnull().sum()}')
print(f'Total number of missing values in the entire DataFrame:\n {df.isnull().sum().sum()}')

# Drop rows with any missing values
df = df.dropna()

# Drop duplicate rows, modifying the DataFrame in place
df = df.drop_duplicates()

print(f'Number of duplicated rows, which should be zero after dropping duplicates:\n {df.duplicated().value_counts()}')

print(f'Summary statistics of your dataset\'s numerical columns:\n {df.describe()}')
print(f'Summary statistics of your dataset\'s categorical columns:')
print(df.describe(include=['object']))

# Encode categorical variables with numeric labels
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Create several binary features based on categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define function to calculate outliers percentage
def outliers_percentage(df):
    outlier_percentages = {}
    outlier_counts = {}
    outliers_combined_df = pd.DataFrame()

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        num_outliers = len(outliers)
        outlier_counts[column] = num_outliers

        total_entries = len(df)
        outlier_percentage = (num_outliers / total_entries) * 100
        outlier_percentages[column] = outlier_percentage

        outliers_combined_df = pd.concat([outliers_combined_df, outliers])

    for col, perc in outlier_percentages.items():
        print(f'{col}: {perc:.2f}%')

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


print(f'Outliers percentage:')
outliers_percentage(df)

# Define target variable
target = 'y'

print(df.head())

# Split into training and validation sets
X = df.drop('y', axis=1)
y = df['y']

scaler = StandardScaler() 
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# AdaBoost model with base DecisionTree
dt_clf = DecisionTreeClassifier(max_depth=1)
ada_clf = AdaBoostClassifier(dt_clf, n_estimators=100, learning_rate=0.1)

# Train the model
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)

# Evaluate classification accuracy
ada_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {ada_accuracy}")

# Hyperparameter tuning for n_estimators and learning_rate
param_range = np.arange(1, 201, 10)
train_scores, test_scores = validation_curve(
    AdaBoostClassifier(dt_clf),
    X_train, y_train,
    param_name="n_estimators",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

# Validation curve
mean_train_score = np.mean(train_scores, axis=1)
mean_test_score = np.mean(test_scores, axis=1)

plt.plot(param_range, mean_train_score, label="Training score")
plt.plot(param_range, mean_test_score, label="Cross-validation score")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("Validation Curve with AdaBoost")
plt.savefig('result/validation_curve_ada_boost.png')
plt.close()

# GradientBoosting model
gradient_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model
gradient_clf.fit(X_train, y_train)
y_pred = gradient_clf.predict(X_test)

gradient_clf = accuracy_score(y_test, y_pred)
print(f"Accuracy: {gradient_clf}")

# Hyperparameter tuning for n_estimators and learning_rate
param_range = np.arange(1, 201, 10)
train_scores, test_scores = validation_curve(
    GradientBoostingClassifier(max_depth=3),
    X_train, y_train,
    param_name="n_estimators",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

# Validation curve
mean_train_score = np.mean(train_scores, axis=1)
mean_test_score = np.mean(test_scores, axis=1)

plt.plot(param_range, mean_train_score, label="Training score")
plt.plot(param_range, mean_test_score, label="Cross-validation score")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("Validation Curve with GradientBoosting")
plt.savefig('result/validation_curve_gradient_boosting.png')
plt.close()

# List of models with increased max_iter for Logistic Regression
models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

# Compare models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T
print(results_df)

# XGBoost model
model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model_xgb.fit(X_train, y_train)

# LightGBM model
model_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model_lgb.fit(X_train, y_train)

# Feature importance analysis
xgb.plot_importance(model_xgb)
plt.title("Feature Importance for XGBoost")
plt.savefig('result/feature_importance_xgboost.png')
plt.close()

lgb.plot_importance(model_lgb)
plt.title("Feature Importance for LightGBM")
plt.savefig('result/feature_importance_lightgbm.png')
plt.close()
