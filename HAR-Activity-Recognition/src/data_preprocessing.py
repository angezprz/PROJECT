import os, glob
import pandas as pd
import numpy as np

def parse_one(file_path: str) -> pd.DataFrame:
    """Load one raw sensor file into a cleaned DataFrame."""
    lower = file_path.lower()
    device = "phone" if os.sep + "phone" + os.sep in lower else "watch"
    sensor = "accel" if os.sep + "accel" + os.sep in lower else "gyro"

    df = pd.read_csv(file_path, header=None, sep=r"[,\s]+", engine="python",
                     usecols=[0,1,2,3,4,5], on_bad_lines="skip")
    df.columns = ["user", "activity", "timestamp", "x", "y", "z"]

    for c in ["x", "y", "z"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(";", ""), errors="coerce")

    df["user"] = pd.to_numeric(df["user"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df["device"] = device
    df["sensor"] = sensor
    df["source_file"] = os.path.basename(file_path)

    return df.dropna(subset=["user","activity","timestamp","x","y","z"])

def load_dataset(base_path: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(base_path, "**", "*.txt"), recursive=True)
    files = [f for f in files if any(x in f.lower() for x in ("accel","gyro"))]

    dfs = []
    for f in files:
        try:
            dfs.append(parse_one(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.astype({"user":"int64","timestamp":"int64","activity":"string"})
    return merged
