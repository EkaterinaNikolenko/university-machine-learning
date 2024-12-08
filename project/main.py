import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

vodafone_music = pd.read_csv('./data/vodafone_music_subset.csv', sep=',')

print(f'First five rows of the vodafone_music:\n {vodafone_music.head()}')
print(f'Summary of the dataframe:\n {vodafone_music.info()}')
print(f'Shape of vodafone_music data:\n {vodafone_music.shape}')

print(f'Number of missing values in each vodafone_music column:\n {vodafone_music.isnull().sum()}')
print(f'Total number of missing values in the entire vodafone_music:\n {vodafone_music.isnull().sum().sum()}')

# Drop rows with any missing values
vodafone_music.fillna(vodafone_music.mean(), inplace=True)

print(f'Number of missing values in each vodafone_music column:\n {vodafone_music.isnull().sum()}')
print(f'Total number of missing values in the entire vodafone_music:\n {vodafone_music.isnull().sum().sum()}')

# Drop duplicate rows, modifying the DataFrame in place
vodafone_music = vodafone_music.drop_duplicates()

print(f'Summary of the dataframe:\n {vodafone_music.info()}')
print(f'Number of duplicated rows, which should be zero after dropping duplicates vodafone_music:\n {vodafone_music.duplicated().value_counts()}')
print(f'Summary statistics of your dataset\'s numerical columns vodafone_music:\n {vodafone_music.describe()}')

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

print(f'Outliers percentage vodafone_music:')
outliers_percentage(vodafone_music)

vodafone_music = vodafone_music.drop(vodafone_music.filter(regex='id|voice|calls|^sms|cost').columns, axis=1)
vodafone_music.info()

# Distribution of target variable values
print(vodafone_music["target"].value_counts(normalize = True))
sns.countplot(x='target', data=vodafone_music)
plt.xlabel('Subscription to the Music service')
plt.ylabel('Count')
plt.title('Distribution of target variable values')
plt.savefig('results/distribution_of_target_variable_values.png')
plt.close()

# Visualize distribution of a few key features
features_to_plot = ['device_type', 'os_category', 'is_my_vf', 'balance_sum', 'paym_last_days', 'lt', 'content_count_m1', 'count_sms_source_5']
for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(vodafone_music[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(f'results/{feature}_distribution.png')
    plt.close()


# Correlation analysis for a subset of features
subset_features = ['device_type', 'os_category', 'is_my_vf', 'balance_sum', 'paym_last_days', 'lt', 'content_count_m1', 'count_sms_source_5', 'target']
correlation_matrix = vodafone_music[subset_features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Selected Features')
plt.savefig('results/correlation_matrix_selected_features.png')
plt.close()


# Pivot table to show average subscription rate by device type and OS category
pivot_table = vodafone_music.pivot_table(values='target', index='device_type', columns='os_category', aggfunc='mean')
print(pivot_table)

# Save pivot table as an image (optional)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
plt.title('Subscription Rate by Device Type and OS Category')
plt.savefig('results/pivot_table_subscription.png')
plt.close()

# Scatter plot for is_my_vf and balance_sum
plt.figure(figsize=(10, 6))
sns.scatterplot(x='balance_sum', y='is_my_vf', hue='target', data=vodafone_music)
plt.title('Balance Sum vs Is My VF')
plt.xlabel('Balance Sum')
plt.ylabel('Is My VF')
plt.savefig('results/balance_sum_vs_is_my_vf.png')
plt.close()

scaler = StandardScaler() 
vodafone_music_scaled = scaler.fit_transform(vodafone_music.drop('target', axis=1))
vodafone_music_scaled = pd.DataFrame(vodafone_music_scaled, columns=vodafone_music.columns[:-1]) 
vodafone_music_scaled['target'] = vodafone_music['target'].values

X = vodafone_music_scaled.drop('target', axis=1)
y = vodafone_music_scaled['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(y_test, y_pred):
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1-score: {f1_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
    print(f"Confusion matrix:\n {confusion_matrix(y_test, y_pred)}")

def plot_validation_curve(model, X_train, y_train, param_name, param_range, cv, scoring='accuracy', model_name='Model', save_path='validation_curve.png'):
    train_scores, val_scores = validation_curve(
        model, X_train, y_train, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )

    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores_mean, label='Training Score', marker='o')
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
    plt.plot(param_range, val_scores_mean, label='Validation Score', marker='o')
    plt.fill_between(param_range, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.2)
    plt.title(f'Validation Curve for {model_name} ({param_name})')
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_top_10_feature_importance(model, X, model_name='Model', save_path='feature_importance.png'):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    top_10_features = feature_importance_df.head(10)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_10_features)
    plt.title(f'Top 10 Feature Importance for {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig(save_path)
    plt.close()

kf = KFold(n_splits=5, shuffle=True, random_state=42)


# K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X_train, y_train)

best_knn = grid_search_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

print("K-Nearest Neighbors:")
evaluate_model(y_test, y_pred_knn)

plot_validation_curve(knn, X_train, y_train, 'n_neighbors', [3, 5, 7, 9, 11], cv=kf, scoring='f1', model_name='K-Nearest Neighbors', save_path='results/validation_curve_knn_n_neighbors_f1.png')
plot_validation_curve(knn, X_train, y_train, 'weights', ['uniform', 'distance'], cv=kf, scoring='f1', model_name='K-Nearest Neighbors', save_path='results/validation_curve_knn_weights_f1.png')
plot_validation_curve(knn, X_train, y_train, 'metric', ['euclidean', 'manhattan'], cv=kf, scoring='f1', model_name='K-Nearest Neighbors', save_path='results/validation_curve_knn_metric_f1.png')


# Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid, cv=kf, scoring='f1', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Random Forest:")
evaluate_model(y_test, y_pred_rf)

plot_top_10_feature_importance(best_rf, X, model_name='Random Forest', save_path='results/top_10_feature_importance_rf.png')
plot_validation_curve(rf, X_train, y_train, 'n_estimators', [50, 100, 150], cv=kf, scoring='f1', model_name='Random Forest', save_path='results/validation_curve_rf_n_estimators_f1.png')
plot_validation_curve(rf, X_train, y_train, 'max_depth', [10, 20, 30], cv=kf, scoring='f1', model_name='Random Forest', save_path='results/validation_curve_rf_max_depth_f1.png')
plot_validation_curve(rf, X_train, y_train, 'min_samples_split', [2, 5, 10], cv=kf, scoring='f1', model_name='Random Forest', save_path='results/validation_curve_rf_min_samples_split_f1.png')


# Gradient Boosting
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7]
}
gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid, cv=kf, scoring='f1', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

best_gb = grid_search_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)

print("Gradient Boosting:")
evaluate_model(y_test, y_pred_gb)

plot_top_10_feature_importance(best_gb, X, model_name='Gradient Boosting', save_path='results/top_10_feature_importance_gb.png')
plot_validation_curve(gb, X_train, y_train, 'n_estimators', [50, 100, 150], cv=kf, scoring='f1', model_name='Gradient Boosting', save_path='results/validation_curve_gb_n_estimators_f1.png')
plot_validation_curve(gb, X_train, y_train, 'learning_rate', [0.01, 0.1, 1], cv=kf, scoring='f1', model_name='Gradient Boosting', save_path='results/validation_curve_gb_learning_rate_f1.png')
plot_validation_curve(gb, X_train, y_train, 'max_depth', [3, 5, 7], cv=kf, scoring='f1', model_name='Gradient Boosting', save_path='results/validation_curve_gb_max_depth_f1.png')


# XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
xgb_model = xgb.XGBClassifier(random_state=42)
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=kf, scoring='f1', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

print("XGBoost:")
evaluate_model(y_test, y_pred_xgb)

plot_top_10_feature_importance(best_xgb, X, model_name='Random Forest', save_path='results/top_10_feature_importance_xgb.png')
plot_validation_curve(xgb_model, X_train, y_train, 'n_estimators', [50, 100, 150], cv=kf, scoring='f1', model_name='XGBoost', save_path='results/validation_curve_xgb_n_estimators_f1.png')
plot_validation_curve(xgb_model, X_train, y_train, 'learning_rate', [0.01, 0.1, 0.2], cv=kf, scoring='f1', model_name='XGBoost', save_path='results/validation_curve_xgb_learning_rate_f1.png')
plot_validation_curve(xgb_model, X_train, y_train, 'max_depth', [3, 5, 7], cv=kf, scoring='f1', model_name='XGBoost', save_path='results/validation_curve_xgb_max_depth_f1.png')

