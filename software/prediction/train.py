import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

DATA_FILE = "flight-data.csv"
MODEL_FILENAME = "altitude_predictor_model.joblib"

try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded data from {DATA_FILE}")
    print(df.head())
except FileNotFoundError:
    print(
        f"Error: {DATA_FILE} not found. Please run the 'Create Dummy Data' script first or ensure your data file is in the correct directory."
    )
    exit()

features = [
    "fluorescenceRaw",
    "uvIndex",
    "outsideTemp",
    "humidity",
]
X = df[features]

y = df["altitude"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("\nTraining the Random Forest Regressor model...")
model.fit(X_train, y_train)
print("Model training complete.")

print("\nEvaluating model performance on the test set...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} meters")
print(f"R-squared (R²): {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Perfect Prediction Line",
)
plt.xlabel("Actual Altitude (meters)")
plt.ylabel("Predicted Altitude (meters)")
plt.title("Actual vs. Predicted Altitude")
plt.grid(True)
plt.legend()
plt.show()

print("\nFeature Importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"- {feature}: {importance:.4f}")

joblib.dump(model, MODEL_FILENAME)
print(f"\nModel saved as {MODEL_FILENAME}")

new_sensor_data = pd.DataFrame(
    [
        {
            "fluorescenceRaw": 35,
            "uvIndex": 7.5,
            "outsideTemp": -25,
            "humidity": 12,
        }
    ]
)

estimated_altitude = model.predict(new_sensor_data)

print(f"\nNew sensor data: {new_sensor_data.iloc[0].to_dict()}")
print(f"Estimated Altitude: {estimated_altitude[0]:.2f} meters")

new_sensor_data_higher = pd.DataFrame(
    [
        {
            "fluorescenceRaw": 15,
            "uvIndex": 10.0,
            "outsideTemp": -50,
            "humidity": 3,
        }
    ]
)

estimated_altitude_higher = model.predict(new_sensor_data_higher)
print(f"\nNew sensor data (higher alt): {new_sensor_data_higher.iloc[0].to_dict()}")
print(f"Estimated Altitude (higher alt): {estimated_altitude_higher[0]:.2f} meters")
