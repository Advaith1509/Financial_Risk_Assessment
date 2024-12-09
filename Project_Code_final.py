# financial_risk_assessment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handling NaN values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
    
    # Converting categorical data into numerical data
    le = LabelEncoder()

    # Encode categorical features
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # le = LabelEncoder()
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         df[col] = le.fit_transform(df[col])

    return df

# Function to train and save models
def train_and_save_models(df):
    # Define features and target
    X = df.drop('Risk Rating', axis=1)  # Replace 'Risk Rating' with your target variable
    y = df['Risk Rating']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Decision Tree model
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, 'decision_tree_model.pkl')

    # Train Logistic Regression model
    lr_model = LogisticRegression(max_iter=10000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'logistic_regression_model.pkl')

    # Train SVM model
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    joblib.dump(svc_model, 'svm_model.pkl')

    # Train Random Forest model with specified parameters
    rf_model = RandomForestClassifier(max_depth=10, max_features='log2', min_samples_split=5, n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'random_forest_model.pkl')

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('financial_risk_assessment.csv')  # Replace with your actual data file path

    # Train and save models
    train_and_save_models(df)
    print("Models trained and saved!")

if __name__ == "__main__":
    main()