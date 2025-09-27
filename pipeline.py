from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def preprocess_data(df, X_cols, y_col, task_type):
    report = {}
    encoders = {}
    missing_before = df.isnull().sum().sum()
    X = df[X_cols].copy()
    y = df[y_col].copy()

    # Handle missing values in X
    for col in X.columns:
        if X[col].dtype in ["int64", "float64"]:
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

    # Handle missing values in y
    if y.isnull().sum() > 0:
        if y.dtype in ["int64", "float64"]:
            y = y.fillna(y.mean())
        else:
            y = y.fillna(y.mode()[0])

    # Encode categorical X columns
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Encode target if classification
    if task_type == "classification" and (
        y.dtype == "object" or str(y.dtype) == "category"
    ):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        encoders[y_col] = le

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Calculate missing after
    missing_after = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1).isnull().sum().sum()
    report["rows_before"] = df.shape[0]
    report["rows_after"] = X_train.shape[0] + X_test.shape[0]
    report["missing_handled"] = missing_before - missing_after
    report["train_shape"] = X_train.shape
    report["test_shape"] = X_test.shape
    report["encoded_columns"] = list(encoders.keys())

    return X_train, X_test, y_train, y_test, report, encoders, scaler

def train_and_evaluate_models(X_train, y_train, X_test, y_test, task_type):
    models = {}
    results = {}
    if task_type == "classification":
        model_list = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVC': SVC(random_state=42)
        }
    else:  # regression
        model_list = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

    for name, model in model_list.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if task_type == "classification":
            results[name] = {
                'Accuracy': accuracy_score(y_test, preds),
                'F1 Score': f1_score(y_test, preds, average='weighted')
            }
        else:
            results[name] = {
                'MSE': mean_squared_error(y_test, preds),
                'R2': r2_score(y_test, preds)
            }
        models[name] = model  # Store fitted model

    return models, results