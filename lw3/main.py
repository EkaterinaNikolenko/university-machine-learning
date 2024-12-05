import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import cross_val_score

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

sns.countplot(x='y', data=df)
plt.xlabel('Subscription to term deposit')
plt.ylabel('Count')
plt.title('Distribution of target variable values')
plt.savefig('plots/distribution_of_target_variable_values.png')
plt.close()

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
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_class = knn_classifier.predict(X_test)

# Evaluate classification accuracy
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Classification accuracy: {accuracy:.2f}")


# Create a generator for cross-validation splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Configure GridSearchCV to optimize n_neighbors
param_grid = {'n_neighbors': np.arange(1, 51)} 
grid_search = GridSearchCV(knn_classifier, param_grid, cv=kf, scoring=make_scorer(accuracy_score)) 
grid_search.fit(X_train, y_train) 

# Best value for n_neighbors
best_k = grid_search.best_params_['n_neighbors'] 
best_score = grid_search.best_score_ 

print(f"Best value for n_neighbors: {best_k}") 
print(f"Best score for the best value: {best_score:.4f}") 

# Plot the metric dependence on n_neighbors
results = grid_search.cv_results_ 
plt.plot(param_grid['n_neighbors'], results['mean_test_score']) 
plt.xlabel('Number of nearest neighbors (k)') 
plt.ylabel('Accuracy score') 
plt.title('Dependence of metric on the number of neighbors (k)') 
plt.savefig('plots/metric_dependence_on_neighbors.png')
plt.close()


# Configure GridSearchCV to find optimal p 
param_grid = {
    'p': np.linspace(1, 10, 10) 
} 

knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance') 
grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy') 
grid_search.fit(X_train, y_train) 
best_p = grid_search.best_params_['p']
best_score = grid_search.best_score_ 
print(f"Best parameters: {best_p}") 
print(f"Best score for the best parameters: {best_score:.4f}")

# Visualize accuracy at different values of p
mean_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

mean_scores = grid_search.cv_results_['mean_test_score'] 
plt.plot(param_grid['p'], mean_scores, label='Accuracy') 
plt.xlabel('Value of parameter p') 
plt.ylabel('Accuracy score') 
plt.title('Dependence of accuracy on parameter p') 
plt.legend() 
plt.savefig('plots/accuracy_vs_parameter_p.png')
plt.close()


# NearestCentroid
centroid_classifier = NearestCentroid()
scores = cross_val_score(centroid_classifier, X_train, y_train, cv=kf, scoring='accuracy')
print(f"Accuracy of NearestCentroid: {scores.mean():.4f}")
