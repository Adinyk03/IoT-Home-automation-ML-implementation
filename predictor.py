import pandas as pd
import joblib
import matplotlib.pyplot as plt

# CONFIGURATION
MODEL_FILE = "simulated_model.pkl"
SCALER_FILE = "simulated_scaler.pkl"
DATA_FILE = "simulated_ina219_data_large.csv"
RATE_PER_KWH = 10  # ₹ per kWh`

# LOAD MODEL & SCALER
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# LOAD DATA
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# FEATURE ENGINEERING
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# SELECT FEATURES & PREDICT
FEATURE_COLUMNS = ['voltage', 'current_mA', 'power_mW', 'hour', 'day_of_week', 'month']
X = df[FEATURE_COLUMNS]
X_scaled = scaler.transform(X)
df['predicted_energy_Wh'] = model.predict(X_scaled)
df['electricity_bill'] = df['predicted_energy_Wh'] * 0.1  # ₹0.10 per Wh

# CATEGORIZE EFFICIENCY BASED ON APPLIANCE TYPE
def categorize_efficiency(device_name, total_energy_Wh):
    appliance_thresholds = {
        'Fridge': 20,  # Example: Fridge efficient if < 20 Wh
        'AC': 50,  # Example: AC efficient if < 50 Wh
        'Phone Charger': 15,  # Phone efficient if < 15 Wh
        'TV': 30,  # TV efficient if < 30 Wh
        'Washing Machine': 40  # Washing machine efficient if < 40 Wh
    }

    threshold = appliance_thresholds.get(device_name, 30)  # Default threshold for unknown devices

    if total_energy_Wh < threshold:
        return 'Efficient'
    elif total_energy_Wh < threshold * 1.5:  # Consider moderate within 1.5 times the threshold
        return 'Moderate'
    else:
        return 'Inefficient'


# AGGREGATE ENERGY CONSUMPTION PER APPLIANCE
appliance_energy = df.groupby('device_name')['predicted_energy_Wh'].sum().reset_index()

# Apply appliance-specific efficiency categorization
appliance_energy['efficiency'] = appliance_energy.apply(
    lambda row: categorize_efficiency(row['device_name'], row['predicted_energy_Wh']),
    axis=1
)

# DISPLAY PREDICTIONS AND EFFICIENCY
print("\n🔍 Device Usage Predictions and Efficiency:")
for _, row in appliance_energy.iterrows():
    print(f"{row['device_name']} ➔ {row['predicted_energy_Wh']:.2f} Wh ➔ ₹{row['predicted_energy_Wh'] * 0.1:.2f} → {row['efficiency']}")

# PLOT 1: Power Usage of All Appliances Over Time (Combined Line Plot)
plt.figure(figsize=(10, 6))
for appliance in df['device_name'].unique():
    grouped = df[df['device_name'] == appliance].groupby('timestamp')['predicted_energy_Wh'].sum().reset_index()
    plt.plot(grouped['timestamp'], grouped['predicted_energy_Wh'], label=appliance)
plt.title("Power Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("Predicted Power Consumption (Wh)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# PLOT 2: Power Usage Breakdown per Appliance (Bar Graph)
plt.figure(figsize=(8, 6))
appliance_energy.set_index('device_name')['predicted_energy_Wh'].plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title("Power Usage Breakdown by Appliance")
plt.xlabel("Appliance")
plt.ylabel("Total Predicted Power Usage (Wh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#PLOT 3: Subplots (One Graph per Appliance with Same Scale)
appliances = df['device_name'].unique()
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

x_min = df['timestamp'].min()
x_max = df['timestamp'].max()
y_min = df['power_mW'].min()
y_max = df['power_mW'].max()

for i, appliance in enumerate(appliances):
    appliance_data = df[df['device_name'] == appliance]
    axes[i].plot(appliance_data['timestamp'], appliance_data['power_mW'], label=f'{appliance}')
    axes[i].set_title(f'{appliance} Power Usage Over Time')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Power (mW)')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    axes[i].legend()
    axes[i].tick_params(axis='x', rotation=45)

# Hide empty subplot if appliances < 6
if len(appliances) < len(axes):
    for j in range(len(appliances), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Assuming 'electricity_bill' column contains the individual bill for each appliance
total_bill = df['electricity_bill'].sum()

# Display the total electricity bill
print(f"\nTotal Estimated Monthly Electricity Bill: ₹{total_bill:.2f}")


# Device Energy Cutoff Thresholds (in Wh)
device_limits = {
    'Phone Charger': 12,  # e.g., Stop charging after 12 Wh (for 10000mAh battery)
    'AC': 50,  # e.g., Stop after 100 Wh (depends on your AC's rating)
    'Fridge': 50,  # Example: Fridge stops after 50 Wh
    'TV': 30,  # Example: TV stops after 30 Wh
    'Washing Machine': 60  # Example: Washing Machine stops after 60 Wh
}

# Function to check if energy consumption exceeds limit
def check_cutoff(device_name, energy_consumed_Wh):
    limit = device_limits.get(device_name, float('inf'))  # Default to no cutoff if not defined
    if energy_consumed_Wh >= limit:
        return True  # Stop the device (cut off)
    return False  # Keep running

# Update the appliance energy aggregation with cutoff logic
appliance_energy = df.groupby('device_name')['predicted_energy_Wh'].sum().reset_index()

# Apply the cutoff check for each appliance
appliance_energy['cutoff_reached'] = appliance_energy.apply(
    lambda row: check_cutoff(row['device_name'], row['predicted_energy_Wh']),
    axis=1
)

# Display whether the appliance should be cut off
print("\n Device Usage Predictions with Cutoff Logic:")
for _, row in appliance_energy.iterrows():
    cutoff_status = "Cutoff" if row['cutoff_reached'] else "Running"
    print(f"{row['device_name']} ➔ {row['predicted_energy_Wh']:.2f} Wh ➔ ₹{row['predicted_energy_Wh'] * 0.1:.2f} → {cutoff_status}")

# You can also apply cutoff logic directly in the data processing steps if you want to simulate stopping consumption:
df['cutoff_reached'] = df.apply(
    lambda row: check_cutoff(row['device_name'], row['predicted_energy_Wh']),
    axis=1
)

# If you want to "stop" the appliance after the threshold, you could set subsequent energy predictions to zero.
df.loc[df['cutoff_reached'], 'predicted_energy_Wh'] = 0

# Plot or display accordingly
plt.figure(figsize=(8, 6))
appliance_energy.set_index('device_name')['predicted_energy_Wh'].plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title(" Power Usage Breakdown by Appliance with Cutoff")
plt.xlabel("Appliance")
plt.ylabel("Total Predicted Power Usage (Wh)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


