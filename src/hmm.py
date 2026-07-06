"""
hmm.py

Defines and trains the Gaussian HMM: 4 hidden states (one per activity),
continuous feature observations, Gaussian emissions, transition matrix
learned via Baum-Welch (hmmlearn's GaussianHMM), decoded with Viterbi.

Model configuration (covariance type, random seed) is chosen using a
validation split carved from TRAINING sessions only, so the held-out
test set is never used for model selection.
"""
import argparse
import os
import pickle
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from preprocess import split_train_test, ACTIVITIES
from features import sessions_to_feature_df, build_sequences

N_STATES = 4


def map_states_to_activities(hidden_states, true_labels):
    """Majority vote: assign each discovered hidden state to its most
    common true activity label."""
    state_to_activity = {}
    for state in np.unique(hidden_states):
        mask = hidden_states == state
        votes = Counter(true_labels[mask])
        state_to_activity[state] = votes.most_common(1)[0][0]
    return state_to_activity


def select_best_config(train_sessions, n_val_per_activity=2, seeds=range(10),
                        cov_types=("diag", "full"), val_seed=7):
    """Try several (covariance_type, seed) combinations, scoring each on a
    validation split carved from training sessions, and return the best.
    The true held-out test set is never touched here."""
    inner_train, val = split_train_test(train_sessions, n_test_per_activity=n_val_per_activity, seed=val_seed)
    inner_train_feat = sessions_to_feature_df(inner_train)
    val_feat = sessions_to_feature_df(val)

    X_it, len_it, y_it, sc = build_sequences(inner_train_feat, fit_scaler=True)
    X_val, len_val, y_val, _ = build_sequences(val_feat, scaler=sc, fit_scaler=False)

    best = {"val_acc": -1}
    for cov_type in cov_types:
        for seed in seeds:
            m = GaussianHMM(n_components=N_STATES, covariance_type=cov_type,
                             n_iter=200, tol=1e-4, random_state=seed)
            m.fit(X_it, lengths=len_it)
            hid_it = m.predict(X_it, lengths=len_it)
            smap = map_states_to_activities(hid_it, y_it)
            hid_val = m.predict(X_val, lengths=len_val)
            pred_val = np.array([smap.get(s, "unknown") for s in hid_val])
            val_acc = np.mean(pred_val == y_val)
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "cov_type": cov_type, "seed": seed}
    return best


def train_final_model(X_train, len_train, y_train, best_cfg):
    """Fit the final GaussianHMM on the full training set via Baum-Welch,
    using the configuration chosen by select_best_config."""
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type=best_cfg["cov_type"],
        n_iter=200,
        tol=1e-4,          # Baum-Welch convergence criterion
        random_state=best_cfg["seed"],
    )
    model.fit(X_train, lengths=len_train)

    train_hidden = model.predict(X_train, lengths=len_train)  # Viterbi
    state_map = map_states_to_activities(train_hidden, y_train)
    return model, state_map


def plot_convergence(model, plots_dir):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(model.monitor_.history, marker='o')
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("Baum-Welch convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "baumwelch_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_transition_matrix(model, state_map, plots_dir):
    labels = [state_map[i] for i in range(model.n_components)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(model.transmat_, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("To state"); ax.set_ylabel("From state")
    ax.set_title("Learned Transition Matrix (A)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{model.transmat_[i, j]:.2f}", ha='center', va='center',
                    color='white' if model.transmat_[i, j] < 0.5 else 'black')
    fig.colorbar(im, ax=ax, label="Transition probability")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "transition_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the Gaussian HMM (Baum-Welch) and decode with Viterbi.")
    parser.add_argument("--proc", default="data/processed")
    parser.add_argument("--plots", default="plots")
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    with open(os.path.join(args.proc, "sessions.pkl"), "rb") as f:
        sess_data = pickle.load(f)
    with open(os.path.join(args.proc, "features.pkl"), "rb") as f:
        feat_data = pickle.load(f)

    train_sessions = sess_data["train"]
    X_train, len_train, y_train = feat_data["X_train"], feat_data["len_train"], feat_data["y_train"]

    print("Selecting HMM configuration via a validation split carved from training data...")
    best_cfg = select_best_config(train_sessions)
    print("Best config:", best_cfg)

    model, state_map = train_final_model(X_train, len_train, y_train, best_cfg)
    print("Converged:", model.monitor_.converged, "| iterations:", len(model.monitor_.history))
    print("Discovered state -> activity mapping:", state_map)

    plot_convergence(model, args.plots)
    plot_transition_matrix(model, state_map, args.plots)

    with open(os.path.join(args.proc, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "state_map": state_map, "best_cfg": best_cfg}, f)
    print(f"Saved trained model to {args.proc}/model.pkl")


if __name__ == "__main__":
    main()
