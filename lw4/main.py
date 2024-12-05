import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Train classification algorithm
dt_clf = DecisionTreeClassifier(criterion="entropy", 
    max_depth=3, 
    random_state=0)
dt_clf.fit(X_train, y_train) 
y_pred = dt_clf.predict(X_test) 

# Evaluate classification accuracy
dt_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {dt_accuracy}")

# Create a generator for cross-validation splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Задання діапазонів гіперпараметрів для пошуку
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Використання GridSearchCV з KFold
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Найкращі параметри
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")

# Отримання результатів
results = grid_search.cv_results_

param_range = [3, 5, 7, 10]
train_scores, valid_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_train,
    y_train,
    param_name="max_depth",
    param_range=[3, 5, 7, 10],
    cv=kf,
    scoring="accuracy"
)

mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Decision Tree Classifier (max_depth)')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_decision_tree_max_depth.png')
plt.close()

param_range = [2, 5, 10]
train_scores, valid_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_train,
    y_train,
    param_name="min_samples_split",
    param_range=param_range,
    cv=kf,
    scoring="accuracy"
)

mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Decision Tree Classifier (min_samples_split)')
plt.xlabel('min_samples_split')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_decision_tree_min_samples_split.png')
plt.close()

param_range = [1, 2, 4]
train_scores, valid_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_train,
    y_train,
    param_name="min_samples_leaf",
    param_range=param_range,
    cv=kf,
    scoring="accuracy"
)

mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Decision Tree Classifier (min_samples_leaf)')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_decision_tree_min_samples_leaf.png')
plt.close()

param_range = ['None', 'sqrt', 'log2']
train_scores, valid_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_train,
    y_train,
    param_name="max_features",
    param_range=[None, 'sqrt', 'log2'],
    cv=kf,
    scoring="accuracy"
)

mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Decision Tree Classifier (max_features)')
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_decision_tree_max_features.png')
plt.close()

def tree_graph_to_png(tree, feature_names, png_file_to_save): 
    tree_str = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None) 
    graph = pydotplus.graph_from_dot_data(tree_str) 
    graph.write_png(png_file_to_save) 
    
tree_graph_to_png(grid_search.best_estimator_, feature_names=X_train.columns, png_file_to_save='result/best_decision_tree.png') 

feature_importances = grid_search.best_estimator_.feature_importances_ 
features = X_train.columns 

importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False) 

plt.figure(figsize=(10, 6)) 
sns.barplot(x='Importance', y='Feature', data=importance_df) 
plt.title('Feature Importance') 
plt.savefig('result/feature_importance.png') 
plt.close()

# Initialize the Random Forest model
rf_clf = RandomForestClassifier(random_state=0)

# Train the model
rf_clf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_clf.predict(X_test)

# Evaluate the accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Hyperparameter grid
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [None, 'sqrt', 'log2']
}

# Initialize GridSearchCV with verbosity to track progress
grid_search_rf = GridSearchCV(estimator=rf_clf, param_grid=param_grid_rf, cv=kf, scoring='accuracy')

# Fit GridSearchCV
grid_search_rf.fit(X_train, y_train)

# Best hyperparameters
print(f"Best parameters found: {grid_search_rf.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")

param_range = [100, 200]
train_scores, valid_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train,
    y_train,
    param_name="n_estimators",
    param_range=param_range,
    cv=kf,
    scoring="accuracy"
)

mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Random Forest Classifier (n_estimators)')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_random_forest_n_estimators.png')
plt.close()

param_range = [3, 5]
train_scores, valid_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train,
    y_train,
    param_name="max_depth",
    param_range=param_range,
    cv=kf,
    scoring="accuracy"
)

mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Random Forest Classifier (max_depth)')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_random_forest_max_depth.png')
plt.close()

param_range = [2, 5]
train_scores, valid_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train,
    y_train,
    param_name="min_samples_split",
    param_range=param_range,
    cv=kf,
    scoring="accuracy"
)

mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Random Forest Classifier (min_samples_split)')
plt.xlabel('min_samples_split')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_random_forest_min_samples_split.png')
plt.close()

# Define hyperparameter range for min_samples_leaf
param_range = [1, 2]
# Compute validation curve
train_scores, valid_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train,
    y_train,
    param_name="min_samples_leaf",
    param_range=param_range,
    cv=kf,
    scoring="accuracy"
)

# Calculate mean and standard deviation
mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Random Forest Classifier (min_samples_leaf)')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_random_forest_min_samples_leaf.png')
plt.close()

# Define hyperparameter range for max_features
param_range = ['None', 'sqrt', 'log2']
# Compute validation curve
train_scores, valid_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X_train,
    y_train,
    param_name="max_features",
    param_range=[None, 'sqrt', 'log2'],
    cv=kf,
    scoring="accuracy"
    # n_jobs=-1
)

# Calculate mean and standard deviation
mean_train_scores = np.mean(train_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)
std_valid_scores = np.std(valid_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, mean_train_scores, label='Training score', color='b')
plt.fill_between(param_range, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.2, color='b')
plt.plot(param_range, mean_valid_scores, label='Validation score', color='g')
plt.fill_between(param_range, mean_valid_scores - std_valid_scores, mean_valid_scores + std_valid_scores, alpha=0.2, color='g')
plt.title('Validation Curve for Random Forest Classifier (max_features)')
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig('result/validation_curve_random_forest_max_features.png')
plt.close()

feature_importances_rf = grid_search_rf.best_estimator_.feature_importances_
features = X_train.columns

importance_df_rf = pd.DataFrame({'Feature': features, 'Importance': feature_importances_rf}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_rf)
plt.title('Top 10 Feature Importance in Random Forest')
plt.savefig('result/feature_importance_rf.png')
plt.close()

# Метод найближчих сусідів
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f"KNN Accuracy: {knn_accuracy}")
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
