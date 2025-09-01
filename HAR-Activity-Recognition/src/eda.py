import matplotlib.pyplot as plt
import pandas as pd

def describe_dataset(df, label_col="activity_label"):
    """Prints basic dataset info: shape, users, activities, devices, sensors."""
    print("Shape:", df.shape)
    print("Users:", df["user"].nunique())
    print("Activities:", df[label_col].nunique())
    print("Devices:", df["device"].unique())
    print("Sensors:", df["sensor"].unique())

def plot_activity_counts(df, label_col="activity_label"):
    counts = df[label_col].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10,4))
    counts.plot(kind="bar")
    plt.title("Samples per Activity")
    plt.ylabel("Rows")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_device_sensor_counts(df):
    combo_counts = (df.groupby(["device","sensor"]).size().reset_index(name="rows"))
    labels = combo_counts["device"] + "_" + combo_counts["sensor"]
    plt.figure(figsize=(6,4))
    plt.bar(labels, combo_counts["rows"])
    plt.title("Rows by Device/Sensor")
    plt.ylabel("Rows")
    plt.show()

def print_activity_pivot(df, label_col="activity_label"):
    pivot_counts = df.pivot_table(index=label_col,
                                  columns=["device","sensor"],
                                  values="x",
                                  aggfunc="count",
                                  fill_value=0).sort_index()
    print(pivot_counts)

def plot_signals(df, activity, user, sensor="accel", n=500):
    sample = df[(df["activity_label"]==activity) & (df["user"]==user) & (df["sensor"]==sensor)].head(n)
    plt.figure(figsize=(10,4))
    plt.plot(sample["timestamp"], sample["x"], label="x")
    plt.plot(sample["timestamp"], sample["y"], label="y")
    plt.plot(sample["timestamp"], sample["z"], label="z")
    plt.legend()
    plt.title(f"Signals ({activity})")
    plt.xlabel("Timestamp")
    plt.ylabel("Acceleration")
    plt.show()
