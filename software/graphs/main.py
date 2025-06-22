import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns

CSV_FILENAME = "rx.csv"
OUTPUT_FILENAME = "stratospore_dashboard_dark.png"

plt.style.use("seaborn-v0_8-darkgrid")

colors = {
    "background": "#2b2b2b",
    "text": "#EAEAEA",
    "grid": "#EAEAEA",
    "outside": "#1e90ff",
    "heater": "#ff4500",
    "pi": "#ffa500",
    "pico": "#20b2aa",
    "fluorescence": "#9370db",
    "uv": "#ffd700",
}

plt.rcParams["text.color"] = colors["text"]
plt.rcParams["axes.labelcolor"] = colors["text"]
plt.rcParams["xtick.color"] = colors["text"]
plt.rcParams["ytick.color"] = colors["text"]


def load_and_prepare_data(filename):
    try:
        df = pd.read_csv(filename)
        df["rxTime"] = pd.to_datetime(df["rxTime"], unit="s")
        df.set_index("rxTime", inplace=True)
        print(f"Successfully loaded {len(df)} records from '{filename}'.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None


def apply_dark_theme_to_axis(ax):
    ax.set_facecolor(colors["background"])

    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.4,
        color=colors["grid"],
        alpha=0.2,
    )

    for spine in ax.spines.values():
        spine.set_color(colors["grid"])
        spine.set_alpha(0.3)


def plot_thermal_management(df, ax):
    """Plots all temperature data on a given axis object."""
    ax.plot(
        df.index,
        df["outsideTemp"],
        label="Outside Temp (°C)",
        color=colors["outside"],
        linewidth=2.5,
        zorder=10,
    )
    ax.plot(
        df.index,
        df["heatingPadTemp"],
        label="Heating Pad Temp (°C)",
        color=colors["heater"],
        linestyle="--",
        linewidth=2.5,
        zorder=5,
    )
    ax.plot(
        df.index,
        df["piTemp"],
        label="Raspberry Pi Temp (°C)",
        color=colors["pi"],
        alpha=0.9,
    )
    ax.plot(
        df.index,
        df["picoTemp"],
        label="Pico Temp (°C)",
        color=colors["pico"],
        alpha=0.9,
    )

    min_temp_time = df["outsideTemp"].idxmin()
    min_temp_val = df["outsideTemp"].min()
    ax.annotate(
        f"Lowest Temp: {min_temp_val:.1f}°C",
        xy=(min_temp_time, min_temp_val),
        xytext=(min_temp_time, min_temp_val + 15),
        arrowprops=dict(facecolor=colors["text"], shrink=0.05, width=1, headwidth=8),
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.7),
    )

    ax.set_title("Thermal System Performance", fontsize=15, weight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.legend(loc="best", frameon=False)


def plot_fluorescence_and_uv(df, ax):
    """Plots fluorescence and UV index on a given axis object with a twin axis."""
    ax.set_ylabel("Fluorescence (Raw)", color=colors["fluorescence"], fontsize=12)
    ax.plot(
        df.index,
        df["fluorescenceRaw"],
        color=colors["fluorescence"],
        label="Fluorescence Raw",
    )
    ax.tick_params(axis="y", labelcolor=colors["fluorescence"])

    max_fluor_time = df["fluorescenceRaw"].idxmax()
    max_fluor_val = df["fluorescenceRaw"].max()
    ax.annotate(
        f"Peak Reading: {max_fluor_val}",
        xy=(max_fluor_time, max_fluor_val),
        xytext=(-75, 25),
        textcoords="offset points",
        arrowprops=dict(facecolor=colors["text"], shrink=0.05, width=1, headwidth=8),
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.7),
    )

    ax2 = ax.twinx()
    ax2.set_ylabel("UV Index", color=colors["uv"], fontsize=12)
    ax2.plot(
        df.index,
        df["uvIndex"],
        color=colors["uv"],
        linestyle=":",
        label="UV Index",
        lw=2,
    )
    ax2.tick_params(axis="y", labelcolor=colors["uv"])
    ax2.grid(False)

    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.set_title("Fluorescence and UV Index", fontsize=15, weight="bold")


def bytes_formatter(x, pos):
    if x < 1024**2:
        return f"{x/1024:.0f} KB"
    return f"{x/1024**2:.1f} MB"


def plot_system_memory(df, ax):
    ax.set_ylabel("Pi Memory (Free)", color=colors["pi"], fontsize=12)
    ax.plot(df.index, df["piMem"], color=colors["pi"], label="Pi Memory (Free)")
    ax.tick_params(axis="y", labelcolor=colors["pi"])
    ax.yaxis.set_major_formatter(FuncFormatter(bytes_formatter))

    ax2 = ax.twinx()
    ax2.set_ylabel("Pico Memory (Free)", color=colors["pico"], fontsize=12)
    ax2.plot(
        df.index,
        df["picoMem"],
        color=colors["pico"],
        linestyle="-.",
        label="Pico Memory (Free)",
    )
    ax2.tick_params(axis="y", labelcolor=colors["pico"])
    ax2.yaxis.set_major_formatter(FuncFormatter(bytes_formatter))
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.set_title("System Memory Stability", fontsize=15, weight="bold")


if __name__ == "__main__":
    sensor_data = load_and_prepare_data(CSV_FILENAME)

    if sensor_data is not None:
        print("Generating dark-theme dashboard...")

        fig, axes = plt.subplots(3, 1, figsize=(15, 22), sharex=True)
        fig.patch.set_facecolor(colors["background"])

        fig.suptitle(
            "Stratospore Ground Test Dashboard", fontsize=22, weight="bold", y=0.98
        )

        for ax in axes:
            apply_dark_theme_to_axis(ax)

        plot_thermal_management(sensor_data, axes[0])
        plot_fluorescence_and_uv(sensor_data, axes[1])
        plot_system_memory(sensor_data, axes[2])

        axes[-1].set_xlabel("Time (HH:MM)", fontsize=14, weight="bold")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        axes[-1].tick_params(axis="x", rotation=0, labelsize=10)

        fig.tight_layout(rect=[0, 0.01, 1, 0.96])

        plt.savefig(
            OUTPUT_FILENAME, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight"
        )
        print(f"Dashboard saved as '{OUTPUT_FILENAME}'")

        plt.show()
