import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

def encode_features(df):
    # Convert 'Attrition' to binary 'Attrition_Yes'
    df['Attrition_Yes'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Drop the original 'Attrition' column
    df.drop('Attrition', axis=1, inplace=True)
    
    # One-hot encode other categorical features
    df = pd.get_dummies(df, drop_first=True).astype('int')
    
    return df
def split_data(df):
    X = df.drop('Attrition_Yes', axis=1)
    y = df['Attrition_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    filepath = "E:/Data Science projects/New folder/IBM_Employee_Attrition/data/HR-Employee-Attrition.csv"

    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    df.to_csv('E:/Data Science projects/New folder/IBM_Employee_Attrition/data/processed_data.csv', index=False)
    # Print the first 5 rows of the processed data
    print(df.head())

    # Print the first 5 rows of the scaled data
    print(X_train_scaled[:5])

    # Print the first 5 rows of the original data
    print(X_train[:5])

    # Print the first 5 rows of the original data
    print(y_train[:5])

    # Print the first 5 rows of the original data
    print(y_test[:5])

    # Print the shapes of the original data
    print("Shape of df:", df.shape)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
    print("Shape of X_train_scaled:", X_train_scaled.shape)
    print("Shape of X_test_scaled:", X_test_scaled.shape)
    print("Data preprocessing successful.")
    
