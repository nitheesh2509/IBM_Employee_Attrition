import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
import pickle
import os
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ensure models folder exists
if not os.path.exists('../models'):
    os.makedirs('../models')

# Load preprocessed data
def load_preprocessed_data(filepath):
    return pd.read_csv(filepath)

# Split the data
def split_data(df):
    X = df.drop('Attrition_Yes', axis=1)
    y = df['Attrition_Yes']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Apply SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

# Train model with grid search (RandomForest)
def build_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'class_weight': ['balanced', None]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Save the model
    with open('../models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    return grid_search.best_estimator_

# Train model with grid search (XGBoost)
def build_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7]
    }
    xgb = XGBClassifier(random_state=42, use_label_encoder=False)
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Save the model
    with open('../models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    return grid_search.best_estimator_

# Evaluate the model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # ROC Curve
    skplt.metrics.plot_roc(y_test, model.predict_proba(X_test), title=f"ROC Curve - {model_name}")
    plt.show()

    return accuracy

# Feature Importance Visualization (for tree-based models)
def plot_feature_importance(model, X_train, model_name):
    importance = model.feature_importances_
    features = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
    features = features.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=features)
    plt.title(f'Feature Importance - {model_name}')
    plt.show()

if __name__ == "__main__":
    # Load data
    filepath = '../data/processed_data.csv'
    df = load_preprocessed_data(filepath)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)

    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(X_train_resampled, X_test)

    # Train and evaluate RandomForest
    rf_model = build_random_forest(X_train_scaled, y_train_resampled)
    rf_accuracy = evaluate_model(rf_model, X_test_scaled, y_test, 'RandomForest')
    plot_feature_importance(rf_model, pd.DataFrame(X_train, columns=X_train.columns), 'RandomForest')

    # Train and evaluate XGBoost
    xgb_model = build_xgboost(X_train_scaled, y_train_resampled)
    xgb_accuracy = evaluate_model(xgb_model, X_test_scaled, y_test, 'XGBoost')
    plot_feature_importance(xgb_model, pd.DataFrame(X_train, columns=X_train.columns), 'XGBoost')

    # Train and evaluate Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=200)
    lr_model.fit(X_train_scaled, y_train_resampled)
    lr_accuracy = evaluate_model(lr_model, X_test_scaled, y_test, 'LogisticRegression')
    
    # Save Logistic Regression model
    with open('../models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    # Train and evaluate GradientBoosting
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train_scaled, y_train_resampled)
    gb_accuracy = evaluate_model(gb_model, X_test_scaled, y_test, 'GradientBoosting')
    
    # Save Gradient Boosting model
    with open('../models/gradient_boosting_model.pkl', 'wb') as f:
        pickle.dump(gb_model, f)

    # Comparison of accuracies with colored bars
    models = ['RandomForest', 'XGBoost', 'LogisticRegression', 'GradientBoosting']
    accuracies = [rf_accuracy, xgb_accuracy, lr_accuracy, gb_accuracy]
    colors = ['blue', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=accuracies, palette=colors)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.show()
