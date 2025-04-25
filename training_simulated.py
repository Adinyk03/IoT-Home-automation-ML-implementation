import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CONFIG
DATA_FILE = "simulated_ina219_data_large.csv"
MODEL_FILE = "simulated_model.pkl"
SCALER_FILE = "simulated_scaler.pkl"
FEATURE_COLUMNS = ['voltage', 'current_mA', 'power_mW', 'hour', 'day_of_week', 'month']
TARGET_COLUMN = 'energy_Wh'

# LOAD DATA
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns before feature engineering: {df.columns.tolist()}")

    # Convert timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    print(f"Columns after feature engineering: {df.columns.tolist()}")
    return df

# PREPROCESS DATA
def preprocess_data(df):
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
    print("\nModel Evaluation:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# SAVE MODEL + SCALER
def save_artifacts(model, scaler):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("\nModel and scaler saved successfully!")

# MAIN
if __name__ == "__main__":
    df = load_dataset(DATA_FILE)
    X_scaled, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, scaler)
