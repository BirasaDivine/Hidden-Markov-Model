"""
evaluate.py

Decodes the held-out, unseen test sessions with Viterbi, then reports a
confusion matrix and per-activity sensitivity / specificity / accuracy.
Also produces the emission-means and decoded-sequence diagnostic plots.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import ACTIVITIES
from features import FEATURE_COLS


def confusion_matrix(y_true, y_pred, labels=ACTIVITIES):
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p not in idx:
            continue
        cm[idx[t], idx[p]] += 1
    return pd.DataFrame(cm, index=labels, columns=labels)


def sensitivity_specificity_table(y_true, y_pred, labels=ACTIVITIES):
    cm = confusion_matrix(y_true, y_pred, labels)
    total = cm.values.sum()
    rows = []
    for lab in labels:
        tp = cm.loc[lab, lab]
        fn = cm.loc[lab].sum() - tp
        fp = cm[lab].sum() - tp
        tn = total - tp - fn - fp
        n_samples = cm.loc[lab].sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        rows.append({"Activity": lab, "Number of Samples": int(n_samples),
                      "Sensitivity": round(sensitivity, 3), "Specificity": round(specificity, 3)})
    overall = np.trace(cm.values) / total
    df = pd.DataFrame(rows)
    df["Overall Accuracy"] = round(overall, 3)
    return df, cm


def plot_confusion_matrix(cm, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, cmap="Blues")
    n = len(cm)
    ax.set_xticks(range(n)); ax.set_xticklabels(cm.columns, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix -- Unseen Test Data")
    vmax = cm.values.max()
    for i in range(n):
        for j in range(n):
            val = cm.values[i, j]
            ax.text(j, i, str(val), ha='center', va='center',
                    color='white' if val > vmax / 2 else 'black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emission_means(model, state_map, plots_dir):
    labels = [state_map[i] for i in range(model.n_components)]
    emission_means = pd.DataFrame(model.means_, index=labels, columns=FEATURE_COLS)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(emission_means.values, cmap='coolwarm', aspect='auto')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xticks(range(len(FEATURE_COLS))); ax.set_xticklabels(FEATURE_COLS, rotation=90, fontsize=8)
    ax.set_title("Emission means per hidden state (Z-scored feature units)")
    fig.colorbar(im, ax=ax, label="Mean (standardized)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "emission_means.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return emission_means


def plot_decoded_sequences(test_feat, test_pred, plots_dir):
    """Plot Viterbi-decoded vs. true activity, one panel per held-out test session."""
    act_to_idx = {a: i for i, a in enumerate(ACTIVITIES)}
    session_ids = sorted(test_feat['session_id'].unique())

    fig, axes = plt.subplots(len(session_ids), 1, figsize=(9, 2 * len(session_ids)))
    if len(session_ids) == 1:
        axes = [axes]

    for ax, sid in zip(axes, session_ids):
        mask = (test_feat['session_id'] == sid).values
        sub_true = test_feat.loc[mask, 'activity'].values
        sub_pred = test_pred[mask]
        true_idx = [act_to_idx[a] for a in sub_true]
        pred_idx = [act_to_idx.get(a, -1) for a in sub_pred]

        ax.step(range(len(true_idx)), true_idx, where='mid', label='True', linewidth=2)
        ax.step(range(len(pred_idx)), pred_idx, where='mid', label='Decoded (Viterbi)', linestyle='--')
        ax.set_yticks(range(len(ACTIVITIES))); ax.set_yticklabels(ACTIVITIES, fontsize=8)
        ax.set_title(sid, fontsize=9)

    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel("Window index (time)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "decoded_sequence_all.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained HMM on held-out unseen test sessions.")
    parser.add_argument("--proc", default="data/processed")
    parser.add_argument("--plots", default="plots")
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    with open(os.path.join(args.proc, "features.pkl"), "rb") as f:
        feat_data = pickle.load(f)
    with open(os.path.join(args.proc, "model.pkl"), "rb") as f:
        model_data = pickle.load(f)

    model, state_map = model_data["model"], model_data["state_map"]
    X_test, len_test, y_test = feat_data["X_test"], feat_data["len_test"], feat_data["y_test"]
    test_feat = feat_data["test_feat"]

    test_hidden = model.predict(X_test, lengths=len_test)  # Viterbi
    test_pred = np.array([state_map[s] for s in test_hidden])

    table, cm = sensitivity_specificity_table(y_test, test_pred)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)
    print("\nEvaluation table:")
    print(table.to_string(index=False))

    plot_confusion_matrix(cm, args.plots)
    plot_emission_means(model, state_map, args.plots)
    plot_decoded_sequences(test_feat, test_pred, args.plots)

    table.to_csv(os.path.join(args.proc, "evaluation_table.csv"), index=False)
    print(f"\nSaved evaluation table and plots to {args.proc} / {args.plots}")


if __name__ == "__main__":
    main()
