#!/usr/bin/env python3
"""
train_probes.py

Train Semantic Entropy Probes (SEPs) from hidden states saved in NPZ files.

Adapted from train-latent-probe.ipynb. Removes the W&B dependency entirely —
data is read from the merged CSV + NPZ files produced by merge_json_csv_hidden_states.py.

Supports:
  - 3 token positions : last_prompt (≈TBG), last (≈SLT), first_answer
  - 3 accuracy labels : LLM_verdict, accuracy_squad_metric, is_correct_string_match
  - In-distribution (ID) and out-of-distribution (OOD) evaluation
  - Per-layer AUROC plots and a single results_summary.csv

Usage:
    python train_probes.py \
        --dataset-dir /path/to/semantic-entropy-probe-dataset \
        --output-dir  ./probe_results
"""

import argparse
import os
import pickle
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 12})
rng = np.random.default_rng(42)


def log(msg, *, ts=False):
    """Print with flush=True (required for nohup). Pass ts=True to prefix timestamp."""
    prefix = f"[{datetime.now().strftime('%H:%M:%S')}] " if ts else ""
    print(f"{prefix}{msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset container
# ─────────────────────────────────────────────────────────────────────────────

class Dataset:
    """
    All data for one QA dataset (e.g. sciq, svamp, trivia_qa).

    hidden_states : dict  token_pos → Tensor(n_layers, n_samples, hidden_dim)
                    Positions:  'last_prompt' ≈ TBG (token before generation)
                                'last'        ≈ SLT (last generated token)
                                'first_answer'  (first generated token, bonus)

    entropy       : Tensor(n_samples,)  raw cluster_assignment_entropy scores
    b_entropy     : Tensor(n_samples,)  binarized entropy — 0=low, 1=high, -1=boundary
                    Set after global best_split() is called across all datasets.

    labels        : dict  col_name → Tensor(n_samples,)
                    Keys: 'LLM_verdict', 'accuracy_squad_metric', 'is_correct_string_match'

    p_true        : Tensor(n_samples,)  p_true baseline value from CSV

    probes        : dict  token_pos → metric → {'models', 'accs', 'aucs'}
                    Populated by train_per_layer_probes().

    final_models  : dict  token_pos → acc_col → {'sep_model', 'ap_model',
                                                   'sep_range', 'ap_range'}
                    Trained on full data for OOD. Populated by train_id_probes().

    id_results    : dict  token_pos → acc_col → {'sep_auc', 'sep_acc', 'ap_auc', 'ap_acc'}
    ood_results   : dict  token_pos → acc_col → {'sep_aucs', 'ap_aucs'}
                    sep_aucs / ap_aucs : dict  target_dataset_name → auroc_dict

    other_ids     : list[int]   indices of the other datasets in Ds
    other_names   : list[str]
    """

    def __init__(self, name, hidden_states, entropy, labels, p_true):
        self.name = name
        self.hidden_states = hidden_states
        self.entropy = entropy
        self.labels = labels
        self.p_true = p_true
        self.b_entropy = None
        self.probes = {}
        self.final_models = {}
        self.id_results = {}
        self.ood_results = {}
        self.other_ids = []
        self.other_names = []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_dir, dataset_name, token_positions, accuracy_cols, model_name="llama"):
    """
    Load one dataset's hidden states and labels from the merged CSV + NPZ files.

    CSV naming convention:
        sep_dataset_hidden_states_{model_name}_{dataset_name}_{token_pos}.csv

    Each CSV has a 'hidden_states_file' column pointing to an NPZ file.
    The NPZ contains 'open_hidden_state_all_layers' with shape
        (n_samples, n_layers, hidden_dim)
    which we transpose to (n_layers, n_samples, hidden_dim) — the shape the
    notebook and all downstream functions expect.

    'hidden_row_index' in the CSV maps each row to its row in the NPZ, so we
    index the NPZ with those values. This handles cases where a CSV is a
    subset of the full NPZ.

    Labels (entropy, accuracy columns, p_true) are identical across all
    token-position CSVs for the same dataset, so we read them once from the
    first CSV encountered.
    """
    hidden_states = {}
    df_labels = None

    for pos in token_positions:
        csv_name = f"sep_dataset_hidden_states_{model_name}_{dataset_name}_{pos}.csv"
        csv_path = os.path.join(dataset_dir, csv_name)
        df = pd.read_csv(csv_path)

        # Capture labels from the first CSV — same across all positions
        if df_labels is None:
            df_labels = df

        npz_path = df["hidden_states_file"].iloc[0]
        log(f"  [{dataset_name}] loading {pos} from {os.path.basename(npz_path)}")
        npz = np.load(npz_path)

        # NPZ layout : (n_samples, n_layers, hidden_dim)
        # Required    : (n_layers, n_samples, hidden_dim)
        hs = torch.tensor(npz["open_hidden_state_all_layers"], dtype=torch.float32)
        hs = hs.permute(1, 0, 2)   # swap sample and layer axes

        # Index with hidden_row_index to select only the rows in this CSV
        row_idx = df["hidden_row_index"].values
        hs = hs[:, row_idx, :]

        hidden_states[pos] = hs
        log(f"         → shape: {tuple(hs.shape)}")

    # ── Labels ────────────────────────────────────────────────────────────
    entropy = torch.tensor(
        df_labels["cluster_assignment_entropy"].values, dtype=torch.float32
    )
    p_true = torch.tensor(df_labels["p_true"].values, dtype=torch.float32)

    labels = {}
    for col in accuracy_cols:
        # astype(float) handles bool columns (LLM_verdict, is_correct_string_match)
        labels[col] = torch.tensor(df_labels[col].astype(float).values, dtype=torch.float32)

    return Dataset(dataset_name, hidden_states, entropy, labels, p_true)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Entropy binarization
# ─────────────────────────────────────────────────────────────────────────────

def best_split(entropy: torch.Tensor, label: str = "dataset") -> float:
    """
    Find the threshold that minimises within-group MSE when splitting the
    continuous entropy distribution into a low-SE group and a high-SE group.

    This is the binarization strategy from Section 4 of the SEP paper
    (ArXiv: 2406.15927). We scan 100 candidate thresholds between a small
    epsilon and max(entropy) and pick the one that minimises the total sum of
    squared deviations from each group's mean.

    A matplotlib curve is plotted so you can visually inspect the choice —
    call plt.show() / save_fig() after calling this for all datasets.
    """
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    mses = []

    for s in splits:
        low = ents[ents < s]
        high = ents[ents >= s]
        if len(low) == 0 or len(high) == 0:
            mses.append(np.inf)
            continue
        mse = np.sum((low - low.mean()) ** 2) + np.sum((high - high.mean()) ** 2)
        mses.append(mse)

    mses = np.array(mses)
    plt.plot(splits, mses, label=label)
    return float(splits[np.argmin(mses)])


def binarize_entropy(entropy: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Convert continuous entropy to binary labels.
        0  → low semantic entropy  (entropy < threshold)
        1  → high semantic entropy (entropy > threshold)
       -1  → boundary sample (entropy == threshold, excluded from probe training)
    """
    binary = torch.full_like(entropy, -1.0)
    binary[entropy < threshold] = 0.0
    binary[entropy > threshold] = 1.0
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bootstrap helpers  (ported from semantic_entropy/uncertainty/utils/eval_utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def auroc_score(y_true, y_score):
    """Compute AUROC from binary labels and predicted probabilities."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    return metrics.auc(fpr, tpr)


def _bootstrap_ci(func, rng, n_resamples=1000):
    """
    Return a function that, given a list of {'y_true', 'y_score'} dicts,
    computes a bootstrapped 90% CI for func using scipy.stats.bootstrap.

    The unusual dict-packing is required because scipy bootstrap passes a
    single iterable to the statistic function rather than two arrays.
    """
    def statistic(samples):
        y_true  = np.array([s["y_true"]  for s in samples])
        y_score = np.array([s["y_score"] for s in samples])
        return func(y_true, y_score)

    def run(y_true, y_score):
        packed = [{"y_true": t, "y_score": s} for t, s in zip(y_true, y_score)]
        result = scipy.stats.bootstrap(
            (packed,), statistic, n_resamples=n_resamples,
            confidence_level=0.90, random_state=rng,
        )
        return {
            "std_err": result.standard_error,
            "low":     result.confidence_interval.low,
            "high":    result.confidence_interval.high,
        }
    return run


def bootstrap_auroc(y_true, y_score):
    """
    Compute AUROC with a bootstrapped 90% CI.
    Returns dict with keys: 'mean', 'bootstrap' (std_err, low, high).
    """
    return {
        "mean":      auroc_score(y_true, y_score),
        "bootstrap": _bootstrap_ci(auroc_score, rng)(y_true, y_score),
    }


def auc_means(aucs: list) -> list:
    """Extract the scalar mean from a list of bootstrap_auroc dicts."""
    return [a["mean"] for a in aucs]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / val / test split helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_split_indices(n_samples, test_frac=0.10, val_frac=0.20, seed=42):
    """
    Split sample indices into train / val / test.

    Mirrors the notebook's two-call train_test_split approach with the same
    random_state so splits are reproducible and identical across layers.

    Fractions (defaults match the notebook):
        test  = 10 % of total
        val   = 20 % of the remaining 90 %  ≈ 18 % of total
        train = remaining                    ≈ 72 % of total
    """
    idx = np.arange(n_samples)
    tv_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=seed)
    train_idx, val_idx = train_test_split(tv_idx, test_size=val_frac, random_state=seed)
    return train_idx, val_idx, test_idx


def get_layer_splits(hs_tensor, labels_np, test_frac=0.10, val_frac=0.20):
    """
    Return per-layer (X_train, X_val, X_test, y_train, y_val, y_test) lists.

    hs_tensor  : Tensor(n_layers, n_samples, hidden_dim)
    labels_np  : np.ndarray(n_samples,)

    All layers share the same sample split (same seed → same indices), so
    there is no data leakage when we later concatenate layers.
    """
    hs_np = hs_tensor.numpy()
    n_samples = hs_np.shape[1]
    train_idx, val_idx, test_idx = make_split_indices(n_samples, test_frac, val_frac)

    X_trains, X_vals, X_tests = [], [], []
    y_trains, y_vals, y_tests = [], [], []

    for layer in range(hs_np.shape[0]):
        X_trains.append(hs_np[layer][train_idx])
        X_vals.append(hs_np[layer][val_idx])
        X_tests.append(hs_np[layer][test_idx])
        y_trains.append(labels_np[train_idx])
        y_vals.append(labels_np[val_idx])
        y_tests.append(labels_np[test_idx])

    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests


def concat_layer_features(hs_tensor, layer_range, idx):
    """
    Concatenate hidden states across a range of layers for a given set of
    sample indices, producing one flat feature vector per sample.

    hs_tensor   : Tensor(n_layers, n_samples, hidden_dim)
    layer_range : (start, end) — half-open interval, same as range(start, end)
    idx         : np.ndarray of sample indices

    Returns np.ndarray(len(idx), (end-start) * hidden_dim)
    """
    start, end = layer_range
    # Select layers and samples, then flatten the layer and feature dims
    selected = hs_tensor[start:end, :, :][:, idx, :]   # (n_sel_layers, len(idx), hidden_dim)
    return selected.permute(1, 0, 2).reshape(len(idx), -1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-layer probe training  (ID – Section 3 of notebook)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_and_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_te):
    """
    Fit a LogisticRegression model and return (val_auroc, test_acc, test_auroc).
    Prints a one-line summary. Used inside the per-layer loop.
    """
    model.fit(X_tr, y_tr)

    val_probs  = model.predict_proba(X_va)
    val_auroc  = roc_auc_score(y_va, val_probs[:, 1])
    val_acc    = np.mean(model.predict(X_va) == y_va)

    test_probs = model.predict_proba(X_te)
    test_auroc = bootstrap_auroc(y_te, test_probs[:, 1])
    test_acc   = np.mean(model.predict(X_te) == y_te)

    log(
        f"    val_acc={val_acc:.3f} val_auroc={val_auroc:.3f} | "
        f"test_acc={test_acc:.3f} test_auroc={test_auroc['mean']:.3f}"
    )
    return val_auroc, test_acc, test_auroc


def train_per_layer_probes(D, token_pos, metric):
    """
    Train one LogisticRegression per transformer layer to predict `metric`
    from the hidden states at `token_pos`.

    metric is either:
        'b_entropy'              → predict binarized semantic entropy
        one of the accuracy cols → predict model correctness

    Boundary samples (b_entropy == -1) are filtered out when training the
    SE probe so the classifier sees only clean 0/1 labels.

    Results stored on D.probes[token_pos][metric]:
        'models' : list of fitted LogisticRegression (one per layer)
        'accs'   : list of float (test accuracy per layer)
        'aucs'   : list of bootstrap_auroc dicts (test AUROC per layer)
    """
    hs = D.hidden_states[token_pos]

    if metric == "b_entropy":
        labels_np = D.b_entropy.numpy()
        # Remove boundary samples — they have no meaningful class label
        mask = labels_np != -1
        hs_use = hs[:, mask, :]
        labels_use = labels_np[mask]
    else:
        hs_use = hs
        labels_use = D.labels[metric].numpy()

    splits = get_layer_splits(hs_use, labels_use)
    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = splits

    models, accs, aucs = [], [], []
    n_layers = hs_use.shape[0]

    for i in range(n_layers):
        log(f"  layer {i + 1:02d}/{n_layers}  ", ts=True)
        model = LogisticRegression(max_iter=1000)
        _, acc, auc = _fit_and_eval(
            model,
            X_trains[i], y_trains[i],
            X_vals[i],   y_vals[i],
            X_tests[i],  y_tests[i],
        )
        models.append(model)
        accs.append(acc)
        aucs.append(auc)

    D.probes.setdefault(token_pos, {})[metric] = {
        "models": models, "accs": accs, "aucs": aucs
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Layer range selection  (Section 3 / decide_layer_range in notebook)
# ─────────────────────────────────────────────────────────────────────────────

def decide_layer_range(Ds, token_pos, metric, min_window=5):
    """
    Find the consecutive layer window that maximises the average test AUROC
    across all datasets for a given (token_pos, metric) combination.

    The same range is then used to build the concatenated-layer final model.
    min_window : the range must span at least this many layers (notebook uses 5).

    Returns (start, end) as a half-open interval.
    """
    # Collect per-layer AUROC means for each dataset
    per_ds = [
        np.array(auc_means(D.probes[token_pos][metric]["aucs"]))
        for D in Ds
    ]
    n_layers = per_ds[0].shape[0]

    best_mean, best_range = -np.inf, (0, min_window)

    for start in range(n_layers):
        for end in range(start + min_window, n_layers + 1):
            mean_auc = np.mean([np.mean(a[start:end]) for a in per_ds])
            if mean_auc > best_mean:
                best_mean, best_range = mean_auc, (start, end)

    log(
        f"  Best range for ({token_pos} / {metric}): "
        f"layers {best_range[0]}–{best_range[1]-1}  avg_AUROC={best_mean:.4f}"
    )
    return best_range


# ─────────────────────────────────────────────────────────────────────────────
# 8. Concatenated-layer ID evaluation  (Section 3 final / train_concat_* in notebook)
# ─────────────────────────────────────────────────────────────────────────────

def train_id_probes(D, token_pos, acc_col, sep_range, ap_range):
    """
    ID evaluation using concatenated hidden states across the best layer range.

    Two probes are trained and tested:
    ① SE Probe (SEP)
        - Trained on binarized entropy (b_entropy) using sep_range layers
        - Boundary samples (b_entropy == -1) excluded from training
        - Tested on (1 - accuracy) because high SE predicts wrongness (error rate)

    ② Accuracy Probe (AP)
        - Trained on the accuracy labels (acc_col) using ap_range layers
        - Tested on the same accuracy labels

    Because both probes use the same random_state=42 for their splits, the
    test indices are consistent and there is no data leakage.

    Both final models are also saved to D.final_models for OOD evaluation
    (trained on full train+val, i.e. everything except the 10% test set).
    """
    hs   = D.hidden_states[token_pos]
    n    = hs.shape[1]
    train_idx, val_idx, test_idx = make_split_indices(n)
    # For final OOD model: train on train+val (all non-test samples)
    trainval_idx = np.concatenate([train_idx, val_idx])

    # ── ① SE Probe ────────────────────────────────────────────────────────
    b_ent = D.b_entropy.numpy()
    mask  = b_ent != -1   # exclude boundary samples

    # Intersect split indices with the non-boundary mask
    mask_set = set(np.where(mask)[0])
    tr_sep   = np.array([i for i in train_idx    if i in mask_set])
    tv_sep   = np.array([i for i in trainval_idx if i in mask_set])

    X_tr_sep  = concat_layer_features(hs, sep_range, tr_sep)
    y_tr_sep  = b_ent[tr_sep]

    sep_id_model = LogisticRegression(max_iter=1000)
    sep_id_model.fit(X_tr_sep, y_tr_sep)

    # Test SEP on error rate (1 - accuracy) using the acc test split
    acc_labels = D.labels[acc_col].numpy()
    X_te_sep   = concat_layer_features(hs, sep_range, test_idx)
    y_te_acc   = acc_labels[test_idx]

    sep_probs  = sep_id_model.predict_proba(X_te_sep)
    sep_auc    = bootstrap_auroc(1 - y_te_acc, sep_probs[:, 1])
    sep_acc    = np.mean(sep_id_model.predict(X_te_sep) == (1 - y_te_acc))
    log(f"  ID SEP   ({token_pos}/{acc_col}): auroc={sep_auc['mean']:.4f}  acc={sep_acc:.4f}")

    # OOD version of SEP: retrain on train+val
    X_tv_sep      = concat_layer_features(hs, sep_range, tv_sep)
    y_tv_sep      = b_ent[tv_sep]
    sep_ood_model = LogisticRegression(max_iter=1000)
    sep_ood_model.fit(X_tv_sep, y_tv_sep)

    # ── ② Accuracy Probe ──────────────────────────────────────────────────
    X_tr_ap   = concat_layer_features(hs, ap_range, train_idx)
    y_tr_ap   = acc_labels[train_idx]
    X_te_ap   = concat_layer_features(hs, ap_range, test_idx)

    ap_id_model = LogisticRegression(max_iter=1000)
    ap_id_model.fit(X_tr_ap, y_tr_ap)

    ap_probs  = ap_id_model.predict_proba(X_te_ap)
    ap_auc    = bootstrap_auroc(y_te_acc, ap_probs[:, 1])
    ap_acc    = np.mean(ap_id_model.predict(X_te_ap) == y_te_acc)
    log(f"  ID AP    ({token_pos}/{acc_col}): auroc={ap_auc['mean']:.4f}  acc={ap_acc:.4f}")

    # OOD version of AP
    X_tv_ap       = concat_layer_features(hs, ap_range, trainval_idx)
    y_tv_ap       = acc_labels[trainval_idx]
    ap_ood_model  = LogisticRegression(max_iter=1000)
    ap_ood_model.fit(X_tv_ap, y_tv_ap)

    # ── Store ──────────────────────────────────────────────────────────────
    D.id_results.setdefault(token_pos, {})[acc_col] = {
        "sep_auc": sep_auc, "sep_acc": sep_acc,
        "ap_auc":  ap_auc,  "ap_acc":  ap_acc,
    }

    D.final_models.setdefault(token_pos, {})[acc_col] = {
        "sep_model": sep_ood_model, "sep_range": sep_range,
        "ap_model":  ap_ood_model,  "ap_range":  ap_range,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. OOD evaluation  (Section 4 of notebook)
# ─────────────────────────────────────────────────────────────────────────────

def ood_eval(Ds, token_pos, acc_col):
    """
    For every dataset D, apply its trained SEP and Acc probe (from final_models)
    to predict accuracy on each of the other datasets.

    The models were trained on D's train+val split. The target datasets are
    evaluated on ALL their samples (no split needed — they are fully OOD).

    SEP predicts error rate, so we test it against (1 - accuracy_labels).
    AP predicts accuracy directly.

    Results stored in D.ood_results[token_pos][acc_col]:
        'sep_aucs' : dict  target_name → bootstrap_auroc dict
        'ap_aucs'  : dict  target_name → bootstrap_auroc dict
    """
    for D in Ds:
        fm        = D.final_models[token_pos][acc_col]
        sep_model = fm["sep_model"]
        ap_model  = fm["ap_model"]
        sep_range = fm["sep_range"]
        ap_range  = fm["ap_range"]

        sep_aucs, ap_aucs = {}, {}

        for j in D.other_ids:
            D_target  = Ds[j]
            hs_target = D_target.hidden_states[token_pos]
            acc_target = D_target.labels[acc_col].numpy()
            all_idx   = np.arange(hs_target.shape[1])

            X_sep = concat_layer_features(hs_target, sep_range, all_idx)
            X_ap  = concat_layer_features(hs_target, ap_range,  all_idx)

            sep_probs = sep_model.predict_proba(X_sep)
            ap_probs  = ap_model.predict_proba(X_ap)

            # SEP trained on error rate → test against error rate
            sep_aucs[D_target.name] = bootstrap_auroc(1 - acc_target, sep_probs[:, 1])
            ap_aucs[D_target.name]  = bootstrap_auroc(acc_target,     ap_probs[:, 1])

            log(
                f"  OOD {D.name} → {D_target.name} ({token_pos}/{acc_col}): "
                f"SEP={sep_aucs[D_target.name]['mean']:.4f}  "
                f"AP={ap_aucs[D_target.name]['mean']:.4f}"
            )

        D.ood_results.setdefault(token_pos, {})[acc_col] = {
            "sep_aucs": sep_aucs, "ap_aucs": ap_aucs
        }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"{name}.png")
    plt.savefig(path, format="png", dpi=150, bbox_inches="tight")
    log(f"  → saved {path}")
    plt.close()


def plot_per_layer_auroc(Ds, token_pos, metric, fig_dir):
    """
    One subplot per dataset showing test AUROC at every transformer layer.
    One figure per (token_pos, metric) combination.
    Lets you see which layers are most informative for a given probe target.
    """
    n = len(Ds)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axs = [axs]

    for D, ax in zip(Ds, axs):
        vals = auc_means(D.probes[token_pos][metric]["aucs"])
        ax.plot(range(1, len(vals) + 1), vals, marker="o", linewidth=2, markersize=4)
        ax.set_title(f"{D.name.upper()} | {token_pos} | {metric}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Test AUROC")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    save_fig(fig_dir, f"per_layer__{token_pos}__{metric}")


def plot_id_barplot(Ds, token_pos, acc_col, fig_dir):
    """
    Bar chart comparing SEP vs Acc probe AUROC for each dataset in the ID setting.
    One figure per (token_pos, acc_col) combination.
    """
    n = len(Ds)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for D, ax in zip(Ds, axes):
        res  = D.id_results[token_pos][acc_col]
        vals = [res["sep_auc"]["mean"], res["ap_auc"]["mean"]]
        bars = ax.bar(["SE Probe", "Acc Probe"], vals, width=0.4)
        ax.set_ylim(0.2, 1.0)
        ax.set_title(f"{D.name.upper()} ID | {token_pos} | {acc_col}")
        ax.set_ylabel("Test AUROC")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_fig(fig_dir, f"id_barplot__{token_pos}__{acc_col}")


def plot_ood_summary(Ds, token_pos, acc_col, fig_dir):
    """
    Stacked comparison of OOD average AUROC (SEP vs Acc probe) per target dataset.
    Averaged across all source datasets that trained on the other datasets.
    """
    # For each target dataset, collect AUROC from all sources that tested on it
    target_sep = defaultdict(list)
    target_ap  = defaultdict(list)

    for D in Ds:
        ood = D.ood_results[token_pos][acc_col]
        for tname, auc in ood["sep_aucs"].items():
            target_sep[tname].append(auc["mean"])
        for tname, auc in ood["ap_aucs"].items():
            target_ap[tname].append(auc["mean"])

    names = list(target_sep.keys())
    sep_means = [np.mean(target_sep[n]) for n in names]
    ap_means  = [np.mean(target_ap[n])  for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w / 2, sep_means, w, label="SE Probe (OOD avg)")
    b2 = ax.bar(x + w / 2, ap_means,  w, label="Acc Probe (OOD avg)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title(f"OOD avg AUROC | {token_pos} | {acc_col}")
    ax.legend()
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig(fig_dir, f"ood_summary__{token_pos}__{acc_col}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Results summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_results_df(Ds, token_positions, accuracy_cols):
    """
    Collect every ID and OOD AUROC into one flat DataFrame with columns:
        train_dataset, test_dataset, token_pos, acc_col, probe_type, setting, auroc

    This lets you compare every combination of:
        - which dataset was used for training vs testing
        - which token position (last_prompt / last / first_answer)
        - which accuracy label (LLM_verdict / accuracy_squad_metric / is_correct_string_match)
        - which probe type (SEP / AccProbe)
        - whether it's ID or OOD
    """
    rows = []

    for D in Ds:
        for token_pos in token_positions:
            for acc_col in accuracy_cols:

                # ID results
                res = D.id_results[token_pos][acc_col]
                for probe, key in [("SEP", "sep_auc"), ("AccProbe", "ap_auc")]:
                    rows.append({
                        "train_dataset": D.name,
                        "test_dataset":  D.name,
                        "token_pos":     token_pos,
                        "acc_col":       acc_col,
                        "probe_type":    probe,
                        "setting":       "ID",
                        "auroc":         res[key]["mean"],
                    })

                # OOD results
                ood = D.ood_results[token_pos][acc_col]
                for tname, auc in ood["sep_aucs"].items():
                    rows.append({
                        "train_dataset": D.name,
                        "test_dataset":  tname,
                        "token_pos":     token_pos,
                        "acc_col":       acc_col,
                        "probe_type":    "SEP",
                        "setting":       "OOD",
                        "auroc":         auc["mean"],
                    })
                for tname, auc in ood["ap_aucs"].items():
                    rows.append({
                        "train_dataset": D.name,
                        "test_dataset":  tname,
                        "token_pos":     token_pos,
                        "acc_col":       acc_col,
                        "probe_type":    "AccProbe",
                        "setting":       "OOD",
                        "auroc":         auc["mean"],
                    })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 12. CLI and main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Semantic Entropy Probes from hidden states (no W&B required)"
    )
    p.add_argument(
        "--dataset-dir", required=True,
        help="Directory containing the merged CSV files from merge_json_csv_hidden_states.py",
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=["sciq", "svamp", "trivia_qa"],
        help="Dataset names to include (must match CSV filename stems)",
    )
    p.add_argument(
        "--token-positions", nargs="+",
        default=["last_prompt", "last", "first_answer"],
        help="Token positions to probe: last_prompt (≈TBG), last (≈SLT), first_answer",
    )
    p.add_argument(
        "--accuracy-cols", nargs="+",
        default=["LLM_verdict", "accuracy_squad_metric", "is_correct_string_match"],
        help="Accuracy column names from the CSV to use as probe targets",
    )
    p.add_argument(
        "--output-dir", default="./probe_results",
        help="Root directory for figures, models, and results_summary.csv",
    )
    p.add_argument(
        "--model-name", default="llama",
        help="Model name for CSV filename pattern (e.g., 'llama', 'qwen')",
    )
    return p.parse_args()


def main():
    args = parse_args()
    fig_dir   = os.path.join(args.output_dir, f"figures/{args.model_name}")
    model_dir = os.path.join(args.output_dir, f"models/{args.model_name}")
    os.makedirs(fig_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── Step 1: Load all datasets into memory ─────────────────────────────
    # All NPZ files are loaded upfront (~48 GB for 3 datasets × 3 positions).
    # With 96 GB available RAM this is safe and avoids repeated disk I/O
    # during the OOD evaluation phase.
    log("\n=== Step 1: Loading datasets ===", ts=True)
    Ds = []
    for name in args.datasets:
        log(f"\n  Dataset: {name}", ts=True)
        D = load_dataset(
            args.dataset_dir, name,
            args.token_positions, args.accuracy_cols,
            model_name=args.model_name,
        )
        Ds.append(D)

    for i, D in enumerate(Ds):
        D.other_ids   = [j for j in range(len(Ds)) if j != i]
        D.other_names = [Ds[j].name for j in D.other_ids]

    # ── Step 2: Binarize entropy with a universal best-split threshold ─────
    # The threshold is found by pooling entropy values from all datasets so
    # the binary labels are comparable across datasets (needed for OOD).
    log("\n=== Step 2: Binarizing entropy ===", ts=True)
    plt.figure()
    all_entropy = torch.cat([D.entropy for D in Ds])
    threshold   = best_split(all_entropy, label="All datasets combined")
    plt.legend()
    plt.title("Within-group MSE at different binarization thresholds")
    plt.xlabel("Threshold"); plt.ylabel("Sum of squared errors")
    save_fig(fig_dir, "entropy_binarization")

    for D in Ds:
        D.b_entropy = binarize_entropy(D.entropy, threshold)
        high_frac   = (D.b_entropy == 1).float().mean().item()
        log(
            f"  {D.name:12s}: threshold={threshold:.4f}  "
            f"high-SE={high_frac:.3f}  dummy_acc={max(high_frac, 1-high_frac):.4f}"
        )

    n_layers = Ds[0].hidden_states[args.token_positions[0]].shape[0]

    # ── Step 3: Per-layer probing ──────────────────────────────────────────
    all_metrics = ["b_entropy"] + args.accuracy_cols
    log("\n=== Step 3: Per-layer probe training ===", ts=True)

    for token_pos in args.token_positions:
        for metric in all_metrics:
            log(f"\n  [{token_pos}] probe target: {metric}", ts=True)
            for D in Ds:
                log(f"    {D.name}", ts=True)
                train_per_layer_probes(D, token_pos, metric)
            plot_per_layer_auroc(Ds, token_pos, metric, fig_dir)

    # ── Step 4: Layer range selection + ID evaluation ──────────────────────
    log("\n=== Step 4: Layer range selection + ID evaluation ===", ts=True)

    for token_pos in args.token_positions:
        for acc_col in args.accuracy_cols:
            log(f"\n  [{token_pos}] acc_col: {acc_col}", ts=True)

            sep_range = decide_layer_range(Ds, token_pos, "b_entropy")
            ap_range  = decide_layer_range(Ds, token_pos, acc_col)

            for D in Ds:
                log(f"    ID eval: {D.name}", ts=True)
                train_id_probes(D, token_pos, acc_col, sep_range, ap_range)

            plot_id_barplot(Ds, token_pos, acc_col, fig_dir)

    # ── Step 5: OOD evaluation ─────────────────────────────────────────────
    log("\n=== Step 5: OOD evaluation ===", ts=True)

    for token_pos in args.token_positions:
        for acc_col in args.accuracy_cols:
            log(f"\n  [{token_pos}] acc_col: {acc_col}", ts=True)
            ood_eval(Ds, token_pos, acc_col)
            plot_ood_summary(Ds, token_pos, acc_col, fig_dir)

    # ── Step 6: Save results summary ──────────────────────────────────────
    log("\n=== Step 6: Saving results ===", ts=True)
    df = build_results_df(Ds, args.token_positions, args.accuracy_cols)
    results_path = os.path.join(args.output_dir, f"results_summary_{args.model_name}.csv")
    df.to_csv(results_path, index=False)
    log(f"\nResults saved → {results_path}")

    pivot = df.pivot_table(
        index=["token_pos", "acc_col", "probe_type"],
        columns="setting",
        values="auroc",
        aggfunc="mean",
    )
    log("\nMean AUROC across datasets:\n")
    log(pivot.to_string())

    # ── Step 7: Save trained models ───────────────────────────────────────
    for D in Ds:
        path = os.path.join(model_dir, f"{D.name}_probes.pkl")
        with open(path, "wb") as f:
            pickle.dump({"name": D.name, "final_models": D.final_models}, f)
        log(f"  Saved model → {path}")

    log("\n=== Done ===", ts=True)


if __name__ == "__main__":
    main()
