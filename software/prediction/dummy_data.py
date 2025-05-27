import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_CSV_FILE = "balloon_sensor_data_full_flight.csv"
PLOTS_DIR = "full_flight_plots"

np.random.seed(42)

num_data_points_per_phase = 500

max_altitude_feet = 100000
max_altitude_meters = max_altitude_feet * 0.3048

time_ascent = np.linspace(0, 10, num_data_points_per_phase)
time_descent = np.linspace(10, 20, num_data_points_per_phase)

altitude_ascent = np.linspace(0, max_altitude_meters, num_data_points_per_phase)
altitude_descent = np.linspace(max_altitude_meters, 0, num_data_points_per_phase)

altitude_profile = np.concatenate((altitude_ascent, altitude_descent))
time_profile = np.concatenate((time_ascent, time_descent))
num_data_points = len(altitude_profile)

temperature = np.zeros(num_data_points)
surface_temp = 20
tropopause_altitude = 11000

lapse_rate_troposphere = -0.0065
lapse_rate_stratosphere_heating = 0.0025

for i, alt in enumerate(altitude_profile):
    if alt <= tropopause_altitude:
        temp = surface_temp + (lapse_rate_troposphere * alt)
    else:
        temp_at_tropopause = surface_temp + (
            lapse_rate_troposphere * tropopause_altitude
        )
        temp = temp_at_tropopause + (
            lapse_rate_stratosphere_heating * (alt - tropopause_altitude)
        )

    temperature[i] = temp + np.random.normal(0, 2.0)

uv_index = 2 + (altitude_profile * 0.00008) + np.random.normal(0, 1.0, num_data_points)
uv_index = np.maximum(0, uv_index)

humidity = np.exp(-altitude_profile / 1500) * 80 + np.random.normal(
    0, 5, num_data_points
)
humidity = np.clip(humidity, 0.1, 95)

# (μW/cm^2)
algae_fluorescence_680nm = 60 + np.random.normal(0, 8, num_data_points)

algae_fluorescence_680nm[temperature < -10] -= (
    -10 - temperature[temperature < -10]
) * 1.8
algae_fluorescence_680nm[uv_index > 9] -= (uv_index[uv_index > 9] - 9) * 5
algae_fluorescence_680nm[humidity < 5] -= (5 - humidity[humidity < 5]) * 0.8
algae_fluorescence_680nm[altitude_profile > 20000] -= (
    altitude_profile[altitude_profile > 20000] - 20000
) * 0.003

algae_fluorescence_680nm = np.maximum(2, algae_fluorescence_680nm)

data = pd.DataFrame(
    {
        "time_index": time_profile,
        "altitude": altitude_profile,
        "algae_fluorescence_680nm": algae_fluorescence_680nm,
        "uv_index": uv_index,
        "temperature_celsius": temperature,
        "humidity_percent": humidity,
    }
)

data.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"Synthetic data for full flight created and saved to '{OUTPUT_CSV_FILE}'")
print(data.head())
print(data.tail())
print(data.describe())

print(f"\nGenerating plots and saving to '{PLOTS_DIR}/'...")

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")

features_to_plot = {
    "altitude": "Altitude (feet)",
    "temperature_celsius": "Temperature (°C)",
    "uv_index": "UV Index",
    "humidity_percent": "Humidity (%)",
    "algae_fluorescence_680nm": r"Algae Fluorescence Irradiance ($μW/cm^2$)",
}

for feature_col, ylabel_text in features_to_plot.items():
    plt.figure(figsize=(14, 8))

    y_values = data[feature_col]
    if feature_col == "altitude":
        y_values = data[feature_col] * 3.28084

    plt.plot(
        data["time_index"], y_values, linestyle="-", color="blue", label=ylabel_text
    )

    if feature_col != "altitude":
        ax2 = plt.gca().twinx()
        ax2.plot(
            data["time_index"],
            data["altitude"] * 3.28084,
            linestyle="--",
            color="red",
            alpha=0.6,
            label="Altitude (feet)",
        )
        ax2.set_ylabel("Altitude (feet)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        plt.legend(loc="upper right")

    plt.title(f"{ylabel_text} vs. Flight Progression (Ascent & Descent)")
    plt.xlabel("Time Index (Arbitrary Units)")
    plt.ylabel(ylabel_text, color="blue" if feature_col != "altitude" else "black")
    plt.tick_params(
        axis="y", labelcolor="blue" if feature_col != "altitude" else "black"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{feature_col}_over_full_flight.png"))
    plt.close()

print("\nGenerating scatter plots against altitude (full profile)...")
for feature_col, ylabel_text in features_to_plot.items():
    if feature_col == "altitude":
        continue
    plt.figure(figsize=(10, 6))
    plt.scatter(data["altitude"] * 3.28084, data[feature_col], alpha=0.4, s=10)
    plt.title(f"{ylabel_text} vs. Altitude (Full Flight Profile)")
    plt.xlabel("Altitude (feet)")
    plt.ylabel(ylabel_text)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"altitude_vs_{feature_col}_scatter_full_flight.png")
    )
    plt.close()

import seaborn as sns

if num_data_points <= 2000:
    print("Generating pair plot (this might take a moment)...")
    pair_plot_fig = sns.pairplot(
        data[
            [
                "altitude",
                "algae_fluorescence_680nm",
                "uv_index",
                "temperature_celsius",
                "humidity_percent",
            ]
        ]
    )
    pair_plot_fig.fig.suptitle("Pair Plot of All Variables (Full Flight)", y=1.02)
    pair_plot_fig.savefig(
        os.path.join(PLOTS_DIR, "pair_plot_all_variables_full_flight.png")
    )
    plt.close()
else:
    print("Skipping pair plot due to large number of data points for performance.")

print(f"Plots saved to the '{PLOTS_DIR}' directory.")
print("The data in 'balloon_sensor_data_full_flight.csv' is ready for ML training.")
