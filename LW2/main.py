import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, make_scorer, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

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

# Filling numerical missing values with mean
df[['age', 'duration', 'campaign', 'pdays', 'previous']] = df[['age', 'duration', 'campaign', 'pdays', 'previous']].fillna(df[['age', 'duration', 'campaign', 'pdays', 'previous']].mean())

# Filling categorical missing values with the most frequent values
df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']] = df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']].apply(lambda x: x.fillna(x.mode()[0]))

# Drop duplicate rows, modifying the DataFrame in place
df = df.drop_duplicates()

print(f'Number of duplicated rows, which should be zero after dropping duplicates:\n {df.duplicated().value_counts()}')

print(f'Summary statistics of your dataset\'s numerical columns:\n {df.describe()}')
print(f'Summary statistics of your dataset\'s categorical columns:')
print(df.describe(include=['object']))

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

print(f'Outliers percentage:')
outliers_percentage(df)

# Scale the numeric columns
scaler = StandardScaler()
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_features] = scaler.fit_transform(df[numeric_features])


# Define target variable
target = 'y'

# Indentify categorical features
categorical_features = df.select_dtypes(include=['object']).columns

# Encode categorical features
for x in categorical_features:
    le = LabelEncoder()
    df[x] = le.fit_transform(df[x])
    print(x, le.classes_)

print(df.head())

# Split data into features and target
X = df.drop(columns=[target])
y = df[target]

# Split data into train, dev and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=200)
model.fit(X_train, y_train)

# Make predictions on dev and test sets
y_train_pred = model.predict(X_train)
y_dev_pred = model.predict(X_dev)
y_test_pred = model.predict(X_test)

# Define function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return f1, precision, recall, accuracy, conf_matrix

# Calculate evaluation metrics for the dev, test set
f1_train, precision_train, recall_train, accuracy_train, conf_matrix_train = calculate_metrics(y_train, y_train_pred)
f1_dev, precision_dev, recall_dev, accuracy_dev, conf_matrix_dev = calculate_metrics(y_dev, y_dev_pred)
f1_test, precision_test, recall_test, accuracy_test, conf_matrix_test = calculate_metrics(y_test, y_test_pred)

# Display results for train and dev sets
print("Development Set Results (train | dev):")
print(f"F1 Score: {f1_train:.2f} | {f1_dev:.2f}")
print(f"Precision: {precision_train:.2f} | {precision_dev:.2f}")
print(f"Recall: {recall_train:.2f} | {recall_dev:.2f}")
print(f"Accuracy: {accuracy_train:.2f} | {accuracy_dev:.2f}")
print("Confusion Matrix (Train):")
print(conf_matrix_train)
print("Confusion Matrix (Dev):")
print(conf_matrix_dev)

# Display results for test set
print("\nTest Set Results:")
print(f"F1 Score: {f1_test:.2f}")
print(f"Precision: {precision_test:.2f}")
print(f"Recall: {recall_test:.2f}")
print(f"Accuracy: {accuracy_test:.2f}")
print("Confusion Matrix (Test):")
print(conf_matrix_test)

# Visualize confusion matrix for test set
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('./confusion_matrix/confusion_matrix.png')
plt.cla()

# Define function to find the best hyperparametrs
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    scorer = make_scorer(r2_score, greater_is_better=True)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

# Hyperparameter settings for different models
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
elasticnet_params = {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9]}

# Findeng the best models and hyperparameters
best_ridge_model, best_ridge_params, ridge_results = hyperparameter_tuning(Ridge(random_state=42), ridge_params, X_train, y_train)
best_lasso_model, best_lasso_params, lasso_results = hyperparameter_tuning(Lasso(random_state=42), lasso_params, X_train, y_train)
best_elasticnet_model, best_elasticnet_params, elasticnet_results = hyperparameter_tuning(ElasticNet(random_state=42), elasticnet_params, X_train, y_train)

# Define the function to calculate RMSE
def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

# Define function to evaluate model quality
def evaluate_model(model, X_dev, y_dev):
    y_pred = model.predict(X_dev)
    mae = mean_absolute_error(y_dev, y_pred)
    rmse_value = rmse(y_dev,  y_pred)
    r2 = r2_score(y_dev, y_pred)
    return mae, rmse_value, r2

# Evaluate the best models
mae_ridge, rmse_ridge, r2_ridge = evaluate_model(best_ridge_model, X_dev, y_dev)
mae_lasso, rmse_lasso, r2_lasso = evaluate_model(best_lasso_model, X_dev, y_dev)
mae_elasticnet, rmse_elasticnet, r2_elasticnet = evaluate_model(best_elasticnet_model, X_dev, y_dev)

# Display the results
print(f"Ridge: MAE={mae_ridge:.2f}, RMSE={rmse_ridge:.2f}, R2={r2_ridge:.2f}")
print(f"Lasso: MAE={mae_lasso:.2f}, RMSE={rmse_lasso:.2f}, R2={r2_lasso:.2f}")
print(f"ElasticNet: MAE={mae_elasticnet:.2f}, RMSE={rmse_elasticnet:.2f}, R2={r2_elasticnet:.2f}")

# Define function to plot validation curves
def plot_validation_curve(param_name, results, param_key, file_name):
    train_scores_mean = results['mean_train_score']
    test_scores_mean = results['mean_test_score']
    param_range = results[f'param_{param_name}']

    plt.figure()
    plt.plot(param_range, train_scores_mean, label="Training score")
    plt.plot(param_range, test_scores_mean, label="Validation score")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.title(f"Validation Curve for {param_key}")
    plt.legend(loc="best")
    plt.savefig(file_name)

# Validation curves for Ridge, Lasso, ElasticNet
plot_validation_curve('alpha', ridge_results, 'Ridge', './validation_curve/ridge_validation_curve.png')
plot_validation_curve('alpha', lasso_results, 'Lasso', './validation_curve/lasso_validation_curve.png')
plot_validation_curve('alpha', elasticnet_results, 'ElasticNet', './validation_curve/elasticnet_validation_curve.png')

# Visualize coefficients of the best model
def plot_coefficients(model, feature_names, title, file_name):
    coef = model.coef_  # Adjusted to work with 1D array if necessary
    if len(coef.shape) == 1:  # If coef_ is 1D
        coef = coef
    else:  # If coef_ is 2D (single response variable), flatten it
        coef = coef[0]
    
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title(title)
    plt.savefig(file_name)

# Get feature names
feature_names = X.columns

# Plot coefficients for the best ElasticNet model (as an example)
plot_coefficients(best_ridge_model, feature_names, 'Ridge Coefficients', './model_coefficients/ridge_model_coefficients.png')
plot_coefficients(best_lasso_model, feature_names, 'Lasso Coefficients', './model_coefficients/lasso_model_coefficients.png')
plot_coefficients(best_elasticnet_model, feature_names, 'ElasticNet Coefficients', './model_coefficients/elasticnet_model_coefficients.png')

