"""
features.py

Extracts time-domain, frequency-domain, and orientation features from
each windowed segment, then assembles per-session feature sequences
(scaled, with lengths) ready for the HMM.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler

from preprocess import TARGET_FS, make_windows

ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
GYR_COLS = ['gyr_x', 'gyr_y', 'gyr_z']
GRAV_COLS = ['grav_x', 'grav_y', 'grav_z']

FEATURE_COLS = [
    'acc_mag_mean', 'acc_mag_std', 'acc_mag_rms', 'acc_mag_min', 'acc_mag_max',
    'gyr_mag_mean', 'gyr_mag_std', 'gyr_mag_rms', 'gyr_mag_min', 'gyr_mag_max',
    'acc_sma', 'gyr_sma', 'acc_corr_xy', 'acc_corr_yz', 'acc_corr_xz',
    'grav_x_mean', 'grav_y_mean', 'grav_z_mean',
    'dominant_freq', 'spectral_energy', 'spectral_entropy',
]


def extract_features(window_df, fs=TARGET_FS):
    """Compute one feature vector (dict) for a single window."""
    feats = {}
    acc = window_df[ACC_COLS].values
    gyr = window_df[GYR_COLS].values
    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)

    # --- Time-domain: intensity and shape of motion ---
    for name, arr in [('acc_mag', acc_mag), ('gyr_mag', gyr_mag)]:
        feats[f'{name}_mean'] = np.mean(arr)
        feats[f'{name}_std'] = np.std(arr)
        feats[f'{name}_rms'] = np.sqrt(np.mean(arr ** 2))
        feats[f'{name}_min'] = np.min(arr)
        feats[f'{name}_max'] = np.max(arr)

    feats['acc_sma'] = np.mean(np.sum(np.abs(acc), axis=1))
    feats['gyr_sma'] = np.mean(np.sum(np.abs(gyr), axis=1))

    for (i, j, name) in [(0, 1, 'acc_corr_xy'), (1, 2, 'acc_corr_yz'), (0, 2, 'acc_corr_xz')]:
        if np.std(acc[:, i]) > 1e-9 and np.std(acc[:, j]) > 1e-9:
            feats[name] = np.corrcoef(acc[:, i], acc[:, j])[0, 1]
        else:
            feats[name] = 0.0

    # --- Orientation: separates postures with similar motion but different phone placement ---
    grav = window_df[GRAV_COLS].values
    feats['grav_x_mean'] = np.mean(grav[:, 0])
    feats['grav_y_mean'] = np.mean(grav[:, 1])
    feats['grav_z_mean'] = np.mean(grav[:, 2])

    # --- Frequency-domain: periodicity of motion ---
    n = len(acc_mag)
    windowed = acc_mag - np.mean(acc_mag)
    fft_vals = np.abs(rfft(windowed))
    freqs = rfftfreq(n, d=1.0 / fs)
    if len(fft_vals) > 1:
        dom_idx = np.argmax(fft_vals[1:]) + 1
        feats['dominant_freq'] = freqs[dom_idx]
        feats['spectral_energy'] = np.sum(fft_vals ** 2) / n
        psd = fft_vals ** 2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        feats['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    else:
        feats['dominant_freq'] = 0.0
        feats['spectral_energy'] = 0.0
        feats['spectral_entropy'] = 0.0

    return feats


def sessions_to_feature_df(sessions):
    """Window every session and extract features, tagging each row with
    its activity label and originating session id (so time order within
    a session can be recovered later)."""
    rows = []
    for name, df in sessions.items():
        activity = df['activity'].iloc[0]
        for w in make_windows(df):
            f = extract_features(w)
            f['activity'] = activity
            f['session_id'] = name
            rows.append(f)
    return pd.DataFrame(rows)


def build_sequences(feat_df, scaler=None, fit_scaler=False):
    """Group feature rows by session (preserving time order) into sequences
    for hmmlearn, which expects a stacked array plus a list of sequence
    lengths. Scaling is fit on training data only."""
    session_ids = feat_df['session_id'].unique()
    X_list, lengths, labels_list = [], [], []
    for sid in session_ids:
        sub = feat_df[feat_df['session_id'] == sid]
        X_list.append(sub[FEATURE_COLS].values)
        lengths.append(len(sub))
        labels_list.append(sub['activity'].values)

    X_raw = np.vstack(X_list)
    if fit_scaler:
        scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)
    labels_flat = np.concatenate(labels_list)
    return X_scaled, lengths, labels_flat, scaler


def main():
    parser = argparse.ArgumentParser(description="Extract windowed features from harmonized sessions.")
    parser.add_argument("--proc", default="data/processed", help="Directory with sessions.pkl / to write features.pkl")
    args = parser.parse_args()

    with open(os.path.join(args.proc, "sessions.pkl"), "rb") as f:
        data = pickle.load(f)
    train_sessions, test_sessions = data["train"], data["test"]

    train_feat = sessions_to_feature_df(train_sessions)
    test_feat = sessions_to_feature_df(test_sessions)
    print("Train feature windows:", train_feat.shape)
    print("Test feature windows:", test_feat.shape)

    X_train, len_train, y_train, scaler = build_sequences(train_feat, fit_scaler=True)
    X_test, len_test, y_test, _ = build_sequences(test_feat, scaler=scaler, fit_scaler=False)

    with open(os.path.join(args.proc, "features.pkl"), "wb") as f:
        pickle.dump({
            "train_feat": train_feat, "test_feat": test_feat,
            "X_train": X_train, "len_train": len_train, "y_train": y_train,
            "X_test": X_test, "len_test": len_test, "y_test": y_test,
            "scaler": scaler,
        }, f)
    print(f"Saved features to {args.proc}/features.pkl")


if __name__ == "__main__":
    main()
