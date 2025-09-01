import numpy as np
import pandas as pd

def harmonize_and_window(df, window_s=5.0, step_s=2.5, purity=0.95, label_col="activity_label"):
    if label_col not in df.columns:
        raise KeyError("Need activity label in dataframe")

    df.sort_values(["user", "device", "sensor", "timestamp"], inplace=True, ignore_index=True)
    df["timestamp"] = df["timestamp"].astype("int64")

    grp_cols = ["user","device","sensor"]
    meta_rows, idx_lists, window_id = [], [], 0

    for key, g in df.groupby(grp_cols, sort=False):
        ts = g["timestamp"].to_numpy()
        n = ts.size
        if n < 2: continue
        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]
        if diffs.size == 0: continue

        dt_ns = np.median(diffs)
        win_n = max(int(round(window_s * 1e9 / dt_ns)), 8)
        step_n = max(int(round(step_s * 1e9 / dt_ns)), 1)

        starts = np.arange(0, n - win_n + 1, step_n, dtype=int)
        base_idx, labels = g.index.to_numpy(), g[label_col].to_numpy()

        for s in starts:
            e = s + win_n
            idx = base_idx[s:e]
            vals, counts = np.unique(labels[s:e], return_counts=True)
            frac = counts.max()/win_n
            if frac < purity: continue
            meta_rows.append({
                "window_id": window_id,
                "user": key[0],
                "device": key[1],
                "sensor": key[2],
                "start_iloc": int(idx[0]),
                "end_iloc": int(idx[-1]) + 1,
                "n_samples": int(win_n),
                "label": vals[counts.argmax()],
                "label_frac": float(frac),
            })
            idx_lists.append(idx)
            window_id += 1
    return pd.DataFrame(meta_rows), idx_lists