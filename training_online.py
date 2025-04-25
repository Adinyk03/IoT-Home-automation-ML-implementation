import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CONFIG
DATA_FILE = "energydata_complete.csv"  # Update path if needed
MODEL_FILE = "online_model.pkl"
SCALER_FILE = "online_scaler.pkl"
FEATURE_COLUMNS = ['T1', 'T2', 'T3', 'RH_1', 'RH_2', 'RH_3', 'T_out', 'Windspeed', 'Visibility', 'Tdewpoint']
TARGET_COLUMN = 'Appliances'

# LOAD DATA
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Dataset file not found!")
        exit(1)

# PREPROCESS DATA
def preprocess_data(df):
    # Check if required columns exist
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# TRAIN MODEL
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# EVALUATE MODEL
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

# SAVE MODEL + SCALER
def save_artifacts(model, scaler):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("\nModel and scaler saved successfully!")

# MAIN
if __name__ == "__main__":
    df = load_dataset(DATA_FILE)
    print("Columns:", df.columns.tolist())

    X_scaled, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, scaler)
