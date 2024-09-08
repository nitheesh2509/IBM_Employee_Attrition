# 1) Import necessary libraries for data handling and preprocessing.
# Import necessary libraries for data handling and preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter

# 2) Loading the Dataset:
# Function to load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Load data
filepath = 'data/HR-Employee-Attrition.csv'
df = load_data(filepath)

# Display the top five rows and shape of the dataset
print("Top five rows of the dataset:\n", df.head())
print("No. of Rows & Columns:", df.shape)


# 3) Data Cleaning:
# Function to clean the data
def clean_data(df):
    # Drop irrelevant columns
    df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df

# Clean the data
df = clean_data(df)

# Check for missing values after cleaning
print("\nNumber of null values after cleaning:\n", df.isnull().sum())


# Function to check class balance of 'Attrition'
def check_class_balance(df):
    print("Class balance for target variable 'Attrition_Yes':")
    print(Counter(df['Attrition']))

# Check class balance
check_class_balance(df)

# 4) Feature Encoding:
# Function to encode features and create 'Attrition_Yes'
def encode_features(df):
    # Convert 'Attrition' to binary 'Attrition_Yes'
    df['Attrition_Yes'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Drop the original 'Attrition' column
    df.drop('Attrition', axis=1, inplace=True)
    
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, drop_first=True).astype('int')
    
    return df

# Encode the features
df = encode_features(df)

# Check the first five rows and shape of the dataset after encoding
print("\nTop five rows after encoding:\n", df.head())
print("Shape of dataset after encoding:", df.shape)


# 5) Split the data:
# Function to split the data
def split_data(df):
    X = df.drop('Attrition_Yes', axis=1)
    y = df['Attrition_Yes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Split the data
X_train, X_test, y_train, y_test = split_data(df)

# Display the shapes of the split data
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# 6) Feature Scaling:
# Function to scale the features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

# Scale the features
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Display the first five rows of scaled X_train
print("\nFirst five rows of scaled X_train:\n", X_train_scaled[:5])
print("\nData Preprocessing Completed")

# 7) Class Balance Check (Optional)
def check_class_balance(df):
    """
    Check the class balance of the target variable 'Attrition_Yes'.
    Args:
        df (pd.DataFrame): Cleaned dataset.
    
    Prints:
        The distribution of classes in the target variable.
    """
    print("Class balance for target variable 'Attrition_Yes':")
    print(Counter(df['Attrition_Yes']))


if __name__ == "__main__":
    # Filepath to the dataset
    filepath = 'data/HR-Employee-Attrition.csv'
    
    # Step 1: Load the data
    df = load_data(filepath)
    print("Step 1: Top five rows of the dataset:\n", df.head())
    print("Step 1: No. of Rows & Columns:", df.shape)

    # Step 2: Clean the data
    df = clean_data(df)
    print("\nStep 2: Number of null values after cleaning:\n", df.isnull().sum())
    
    # Step 3: Check class balance for the target variable 'Attrition_Yes'
    check_class_balance(df)
    
    # Step 4: Encode categorical features
    df = encode_features(df)
    print("\nStep 4: Top five rows after encoding:\n", df.head())
    print("Step 4: Shape of dataset after encoding:", df.shape)

    # Step 5: Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df)
    print("\nStep 5: Shape of X_train:", X_train.shape)
    print("Step 5: Shape of X_test:", X_test.shape)
    print("Step 5: Shape of y_train:", y_train.shape)
    print("Step 5: Shape of y_test:", y_test.shape)
    
    # Step 6: Scale the features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("\nStep 6: Feature scaling completed.")
    print("Step 6: First five rows of scaled X_train:\n", X_train_scaled[:5])

    print("\nData Preprocessing Completed")
