"""
preprocess.py

Loads raw sensor CSVs, harmonizes 50Hz/100Hz recordings to a common 50Hz
grid, sets aside an unseen test split, and segments recordings into
overlapping windows for feature extraction.
"""
import argparse
import glob
import os
import pickle
import random
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import decimate

TARGET_FS = 50
ACTIVITIES = ['standing', 'walking', 'jumping', 'still']
SIGNAL_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
               'grav_x', 'grav_y', 'grav_z']

WINDOW_SEC = 2.0
OVERLAP = 0.5
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_FS)             # 100
STRIDE_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))     # 50


def load_and_harmonize(raw_dir, target_fs=TARGET_FS):
    """Load every CSV in raw_dir, decimating 100Hz recordings down to
    target_fs (50Hz) with an anti-aliasing filter. Returns a dict of
    DataFrames keyed by session (file) name."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    sessions = {}

    for f in files:
        df = pd.read_csv(f)
        fname = os.path.basename(f).replace(".csv", "")
        fs = int(df['fs_hz'].iloc[0])
        activity = df['activity'].iloc[0]

        if fs == target_fs:
            df_out = df[SIGNAL_COLS].copy()
        elif fs > target_fs and fs % target_fs == 0:
            factor = fs // target_fs
            decimated = {col: decimate(df[col].values, factor, zero_phase=True)
                         for col in SIGNAL_COLS}
            df_out = pd.DataFrame(decimated)
        else:
            raise ValueError(f"Unexpected fs={fs} in {fname}, cannot cleanly decimate to {target_fs}")

        df_out['activity'] = activity
        df_out['fs_hz'] = target_fs
        df_out['session_id'] = fname
        sessions[fname] = df_out

    return sessions


def split_train_test(sessions, n_test_per_activity=2, seed=42):
    """Hold out n_test_per_activity complete sessions per activity as an
    unseen test set. These sessions are never used for training, scaling,
    or model selection."""
    rng = random.Random(seed)
    by_activity = defaultdict(list)
    for name, df in sessions.items():
        by_activity[df['activity'].iloc[0]].append(name)

    test_names = set()
    for act, names in by_activity.items():
        chosen = rng.sample(sorted(names), n_test_per_activity)
        test_names.update(chosen)

    train_sessions = {k: v for k, v in sessions.items() if k not in test_names}
    test_sessions = {k: v for k, v in sessions.items() if k in test_names}
    return train_sessions, test_sessions


def make_windows(df, window=WINDOW_SAMPLES, stride=STRIDE_SAMPLES):
    """Segment a session's samples into overlapping fixed-length windows."""
    n = len(df)
    return [df.iloc[start:start + window] for start in range(0, n - window + 1, stride)]


def plot_sample_signals(sessions, plots_dir, activities=ACTIVITIES):
    """Plot acceleration magnitude over time for one representative
    recording of each activity -- a quick sanity check before modeling."""
    fig, axes = plt.subplots(len(activities), 1, figsize=(9, 8))
    for ax, act in zip(axes, activities):
        sample = next(df for df in sessions.values() if df['activity'].iloc[0] == act)
        t = np.arange(len(sample)) / TARGET_FS
        acc_mag = np.linalg.norm(sample[['acc_x', 'acc_y', 'acc_z']].values, axis=1)
        ax.plot(t, acc_mag)
        ax.set_title(f"{act} -- acceleration magnitude")
        ax.set_ylabel("m/s^2")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "sample_signals.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Load, harmonize, split, and window raw recordings.")
    parser.add_argument("--raw", default="data/raw", help="Directory of raw labelled CSV clips")
    parser.add_argument("--out", default="data/processed", help="Directory to write processed pickles")
    parser.add_argument("--plots", default="plots", help="Directory to write diagnostic plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    sessions = load_and_harmonize(args.raw)
    print(f"Loaded and harmonized {len(sessions)} recording sessions")

    plot_sample_signals(sessions, args.plots)

    train_sessions, test_sessions = split_train_test(sessions)
    print(f"Train sessions: {len(train_sessions)} | Held-out test sessions: {len(test_sessions)}")
    print("Held-out test files:", sorted(test_sessions.keys()))

    with open(os.path.join(args.out, "sessions.pkl"), "wb") as f:
        pickle.dump({"train": train_sessions, "test": test_sessions}, f)

    print(f"Saved harmonized sessions to {args.out}/sessions.pkl")


if __name__ == "__main__":
    main()
