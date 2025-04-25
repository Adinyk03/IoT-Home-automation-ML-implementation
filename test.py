import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------
# Load and Prepare Dataset
# ----------------------
df = pd.read_csv("energydata_complete.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.drop(columns=['rv1', 'rv2'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df = df.drop(columns=['date'])

X = df.drop(columns=['Appliances'])
y = df['Appliances']

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

joblib.dump(model, "energy_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Model and scaler saved successfully!")

# ----------------------
# Simulated Device Usage Predictions (with auto-fix for missing columns)
# ----------------------
model = joblib.load("energy_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = X.columns.tolist()
expected_feature_count = len(feature_names)

# Device Simulation Data (some rows have 27 values)
devices = ["AC", "Fridge", "TV", "Washing Machine", "Phone Charger"]
device_inputs = [
    [22, 35, 21.5, 21.0, 20.5, 23.0, 24.0, 22.5, 23.5, 48.0, 50.0, 55.0, 52.0, 49.0, 48.0, 50.5, 51.5, 10.0, 75.0, 1013.0, 3.0, 40.0, 5.0, 10, 3, 7],  # AC
    [4, 6, 21.0, 20.5, 21.0, 23.0, 23.5, 22.5, 23.5, 42.0, 45.0, 47.0, 46.0, 44.0, 43.0, 45.5, 46.5, 5.0, 50.0, 1012.0, 2.0, 30.0, 3.0, 10, 3, 7],      # Fridge
    [19, 12, 22.0, 21.0, 22.5, 25.0, 26.0, 25.5, 26.0, 50.0, 55.0, 60.0, 57.0, 52.0, 50.0, 55.0, 56.0, 15.0, 70.0, 1011.0, 3.0, 35.0, 4.0, 10, 3, 7],  # TV
    [5, 14, 22.0, 21.5, 20.0, 24.0, 25.0, 23.0, 24.0, 55.0, 56.0, 58.0, 54.0, 51.0, 50.0, 53.0, 54.0, 20.0, 65.0, 1014.0, 3.5, 33.0, 6.0, 10, 3, 7],   # Washing Machine
    [20, 22, 20.0, 20.5, 20.0, 21.0, 22.0, 21.5, 21.0, 35.0, 36.0, 38.0, 37.0, 36.0, 35.5, 36.0, 36.5, 2.0, 30.0, 1015.0, 1.0, 25.0, 2.0, 10, 3, 7]     # Charger
]

print("\n🔍 Device Usage Predictions:")

for i, row in enumerate(device_inputs):
    # Auto-fix: Pad row if it's too short
    while len(row) < expected_feature_count:
        row.append(0.0)

    df_input = pd.DataFrame([row], columns=feature_names)
    scaled_input = scaler.transform(df_input)
    prediction = model.predict(scaled_input)[0]
    predicted_kwh = prediction / 1000
    cost = predicted_kwh * 16  # ₹ per kWh

    suggestion = ""
    if prediction > 400:
        suggestion = "⚠️ High usage — consider limiting use."
    elif prediction < 100:
        suggestion = "✅ Efficient usage."
    else:
        suggestion = "🟡 Moderate usage."

    print(f"{devices[i]} ➤ {prediction:.2f} Wh ➤ ₹{cost:.2f} → {suggestion}")


import matplotlib.pyplot as plt

# Device predictions already exist from earlier
devices = ['AC', 'Fridge', 'TV', 'Washing Machine', 'Phone Charger']
predicted_wh = [273.2, 272.9, 271.3, 287.1, 299.0]  # Replace with your actual predicted values
rate_per_kwh = 16  # ₹/kWh

# Compute per-device cost and total
predicted_kwh = [val / 1000 for val in predicted_wh]
costs = [round(kwh * rate_per_kwh, 2) for kwh in predicted_kwh]
total_cost = sum(costs)

# Print Total Daily Estimate
print(f"\n💸 Total Estimated Daily Electricity Bill: ₹{total_cost:.2f}\n")

# Plot Bar Chart
plt.figure(figsize=(8, 5))
bars = plt.bar(devices, predicted_wh, color='skyblue')
plt.title("📊 Device-wise Energy Consumption (Wh)")
plt.ylabel("Energy (Wh)")
plt.xlabel("Devices")

# Annotate each bar
for bar, val in zip(bars, predicted_wh):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.1f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 🔋 Simulate AI Optimization (Phone Charging Control)
phone_usage = predicted_wh[-1]  # Last device is phone
charge_limit_wh = 280  # Example: stop charging if it goes beyond this
if phone_usage > charge_limit_wh:
    print("Phone charging exceeded safe limit. Automatically stopping charge to prevent overuse. ✅")
else:
    print("Phone charging within safe limit. No action needed.")
