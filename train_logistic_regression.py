import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model(file_path):
    print(f"Loading dataset from {file_path}...")
    # Load data
    df = pd.read_excel(file_path, sheet_name='Loan_Dataset')

    # Preprocessing
    # Drop unnecessary demo column
    if 'default_timing_demo_only' in df.columns:
        df = df.drop(columns=['default_timing_demo_only'])

    # Encode target variable
    df['loan_outcome'] = df['loan_outcome'].map({'Approved': 1, 'Rejected': 0})

    # Encode categorical feature
    df = pd.get_dummies(df, columns=['employment_type'], drop_first=True)

    # Define X and y
    X = df.drop(columns=['loan_outcome'])
    y = df['loan_outcome']

    # 80/20 Train-test split
    print("Splitting data into 80% training and 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # Predictions
    print("Evaluating model...")
    y_pred = log_reg.predict(X_test_scaled)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # Feature importance (coefficients)
    feature_names = X.columns
    coefficients = log_reg.coef_[0]
    importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
    importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False)
    
    print("-" * 30)
    print("Feature Coefficients (Influence on Approval):")
    print(importance_df[['Feature', 'Coefficient']])
    
    return log_reg, scaler

if __name__ == "__main__":
    dataset_file = "bharat_horizon_home_loan_dataset_v2.xlsx"
    train_model(dataset_file)
