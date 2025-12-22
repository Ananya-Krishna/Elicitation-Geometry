#!/usr/bin/env python3
"""
Geometric Alignment Analysis for All Models (llama2, falcon, mistral)

This script (for each model_family in {llama2, falcon, mistral}):

  1. Loads SAE encodings:
     - {base_path}/new_approach/sae_output_ultra/{model}_base_sae_encoded.npy
     - {base_path}/new_approach/sae_output_ultra/{model}_aligned_sae_encoded.npy

  2. Loads Neural-ODE actions:
     - {base_path}/new_approach/ode_output_ultra/{model}/{model}_actions.npy

  3. Loads toxicity scores from RealToxicityPrompts-style JSONs:
     - {data_dir}/prompts_split_*.json
     - Uses only the first N = encodings.shape[0] samples.

  4. Normalizes SAE space (shared mean/std for base+aligned).

  5. Computes:
     - Global Wasserstein-2 OT distance W2(base_norm, aligned_norm)
     - Per-sample geodesic_sq = ||z_aligned_norm - z_base_norm||^2
     - Per-sample ratio = action / geodesic_sq

  6. Toxicity-binned analysis:
     - Bins: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.01]
     - For each bin:
         * W2_bin between base & aligned restricted to that bin
         * mean_action_bin
         * mean_geodesic_sq_bin
         * mean_ratio_bin

  7. Alignment subspace:
     - Δz = z_aligned_norm - z_base_norm
     - PCA on Δz, keep top 3 PCs
     - Correlate PC1 scores with toxicity and action

  8. t-SNE visualization (for intuition):
     - Combine base_norm and aligned_norm
     - PCA → 50D → t-SNE → 2D
     - Plots:
         * Base vs aligned
         * Base colored by action

  9. Saves per-model outputs:
     - {output_dir}/{model}_geom_analysis.json
     - {output_dir}/{model}_global_stats.csv
     - {output_dir}/{model}_toxicity_bins.csv
     - Figures in {output_dir}/figures/{model}/

 10. Saves combined outputs:
     - {output_dir}/all_models_summary.json
     - {output_dir}/all_models_global_stats.csv
     - Cross-model bar plots in {output_dir}/figures/

Requirements:
  - pip install pot scikit-learn matplotlib
"""

import argparse
import csv
import json
from pathlib import Path
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import ot  # Python Optimal Transport
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# -----------------------------------------------------------------------------
# UTILS: LOADING
# -----------------------------------------------------------------------------

def load_encodings(sae_dir: Path, model_family: str):
    # base_path = sae_dir / f"{model_family}_base_sae_encoded.npy"
    # aligned_path = sae_dir / f"{model_family}_aligned_sae_encoded.npy"
    base_path = sae_dir / f"{model_family}_base_sae_encoded_first_half.npy"
    aligned_path = sae_dir / f"{model_family}_aligned_sae_encoded_first_half.npy"

    if not base_path.exists() or not aligned_path.exists():
        raise FileNotFoundError(f"Missing encodings for {model_family}: "
                                f"{base_path}, {aligned_path}")

    base = np.load(base_path)
    aligned = np.load(aligned_path)

    if base.shape != aligned.shape:
        raise ValueError(f"Shape mismatch for {model_family}: "
                         f"base {base.shape}, aligned {aligned.shape}")

    print(f"[{model_family}] Loaded encodings: {base.shape}")
    return base.astype(np.float32), aligned.astype(np.float32)


def load_actions(ode_dir: Path, model_family: str, n_samples: int):
    actions_path = ode_dir / model_family / f"{model_family}_actions.npy"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions file for {model_family}: {actions_path}")

    actions = np.load(actions_path).astype(np.float32)
    if actions.shape[0] < n_samples:
        raise ValueError(
            f"[{model_family}] actions length {actions.shape[0]} < encodings {n_samples}"
        )

    actions = actions[:n_samples]
    print(f"[{model_family}] Loaded actions: {actions.shape}")
    return actions


def load_toxicity_scores(data_dir: Path, n_samples: int):
    """
    Load toxicity scores from RealToxicityPrompts-style JSONs:
      data_dir/prompts_split_0.json, ..., prompts_split_5.json

    Assumes encodings correspond to the first n_samples prompts in this ordering.
    """
    all_tox = []

    for i in range(6):
        fpath = data_dir / f"prompts_split_{i}.json"
        if not fpath.exists():
            print(f"[toxicity] Warning: {fpath} not found, skipping")
            continue
        with open(fpath, "r") as f:
            prompts = json.load(f)
        for p in prompts:
            all_tox.append(float(p.get("toxicity", 0.0)))

    all_tox = np.array(all_tox, dtype=np.float32)
    # After tox_all = np.array(tox_all, dtype=float)
    print("DEBUG global tox range:", all_tox.min(), all_tox.max())
    global_hist, _ = np.histogram(all_tox, bins=[0, 0.25, 0.5, 0.75, 1.0])
    print("DEBUG global counts [0–.25, .25–.5, .5–.75, .75–1]:", global_hist)

    subset = all_tox[:n_samples]
    subset_hist, _ = np.histogram(subset, bins=[0, 0.25, 0.5, 0.75, 1.0])
    print(f"DEBUG subset (first {n_samples}) counts:", subset_hist)
    print("DEBUG first 10 subset tox values:", subset[:10])
    return subset


    # if all_tox.shape[0] < n_samples:
    #     raise ValueError(
    #         f"Toxicity entries {all_tox.shape[0]} < encodings {n_samples}. "
    #         f"Check data/encodings alignment."
    #     )

    # tox = all_tox[:n_samples]
    # print(f"[toxicity] Loaded {tox.shape[0]} toxicity scores (using first {n_samples})")
    # return tox


# -----------------------------------------------------------------------------
# GEOMETRY: NORMALIZATION, OT, GEODESIC
# -----------------------------------------------------------------------------

def normalize_features(base: np.ndarray, aligned: np.ndarray):
    stacked = np.vstack([base, aligned])
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-8

    base_norm = (base - mean) / std
    aligned_norm = (aligned - mean) / std

    return base_norm.astype(np.float32), aligned_norm.astype(np.float32)


def compute_w2(base_norm: np.ndarray, aligned_norm: np.ndarray, reg: float = 0.01):
    """
    Compute Sinkhorn-regularized Wasserstein-2 distance between two
    empirical distributions with equal weights.
    """
    n = base_norm.shape[0]
    assert n == aligned_norm.shape[0]

    print(f"    [OT] Computing cost matrix ({n} x {n})...")
    M = ot.dist(base_norm, aligned_norm, metric="euclidean")

    a = np.ones(n) / n
    b = np.ones(n) / n

    print(f"    [OT] Running Sinkhorn2...")
    w2 = ot.sinkhorn2(a, b, M, reg=reg)

    return float(w2)


# -----------------------------------------------------------------------------
# PLOTTING HELPERS (PER MODEL)
# -----------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_action_hist(model_family, actions, fig_dir: Path):
    plt.figure(figsize=(6, 4))
    plt.hist(actions, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Action A(x)")
    plt.ylabel("Count")
    plt.title(f"{model_family}: Action Distribution")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_action_hist.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")


def plot_action_vs_toxicity(model_family, actions, toxicity, fig_dir: Path):
    plt.figure(figsize=(6, 4))
    plt.scatter(toxicity, actions, s=8, alpha=0.5)
    plt.xlabel("Toxicity score")
    plt.ylabel("Action A(x)")
    plt.title(f"{model_family}: Action vs Toxicity")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_action_vs_toxicity_scatter.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")


def plot_action_by_tox_bin(model_family, actions, toxicity, fig_dir: Path):
    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
    data = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (toxicity >= lo) & (toxicity < hi)
        if mask.sum() > 0:
            data.append(actions[mask])
        else:
            data.append(np.array([]))

    plt.figure(figsize=(6, 4))
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.7)

    plt.xlabel("Toxicity bin")
    plt.ylabel("Action A(x)")
    plt.title(f"{model_family}: Action by Toxicity Bin")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_action_by_toxicity_bin_boxplot.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")


def plot_ratio_hist(model_family, ratio, fig_dir: Path):
    plt.figure(figsize=(6, 4))
    # Clip extreme ratios for visualization
    clipped = np.clip(ratio, 0, np.percentile(ratio, 99))
    plt.hist(clipped, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Action / Geodesic^2")
    plt.ylabel("Count")
    plt.title(f"{model_family}: Curvature Ratio Distribution")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_ratio_action_over_geod_hist.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")


def plot_tox_bin_w2_and_action(model_family, bin_stats, fig_dir: Path):
    labels = []
    w2_vals = []
    act_vals = []

    for label, stats in bin_stats.items():
        if stats.get("skipped", False):
            continue
        labels.append(label)
        w2_vals.append(stats["w2"])
        act_vals.append(stats["mean_action"])

    if not labels:
        print(f"    [fig] Skipping tox-bin W2/action plot for {model_family}: no bins.")
        return

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, w2_vals, width=width, label="W2")
    plt.bar(x + width / 2, act_vals, width=width, label="Mean Action")
    plt.xticks(x, labels, rotation=30)
    plt.ylabel("Value")
    plt.title(f"{model_family}: W2 and Mean Action by Toxicity Bin")
    plt.legend()
    plt.tight_layout()
    out = fig_dir / f"{model_family}_tox_bin_w2_and_mean_action.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")


def plot_pca_delta(model_family, delta_pca, toxicity, actions, explained, fig_dir: Path):
    # Explained variance
    plt.figure(figsize=(5, 3.5))
    x = np.arange(len(explained))
    plt.bar(x, explained)
    plt.xticks(x, [f"PC{i+1}" for i in x])
    plt.ylabel("Explained variance ratio")
    plt.title(f"{model_family}: Δz PCA Explained Variance")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_delta_pca_explained_variance.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")

    # PC1 vs PC2 colored by toxicity
    pc1 = delta_pca[:, 0]
    pc2 = delta_pca[:, 1]

    plt.figure(figsize=(6, 4))
    sc = plt.scatter(pc1, pc2, c=toxicity, s=8, alpha=0.7, cmap="viridis")
    plt.colorbar(sc, label="Toxicity")
    plt.xlabel("PC1(Δz)")
    plt.ylabel("PC2(Δz)")
    plt.title(f"{model_family}: Δz PCA (PC1 vs PC2) colored by toxicity")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_delta_pca_pc1_pc2_toxicity.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")

    # PC1 vs PC2 colored by action
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(pc1, pc2, c=actions, s=8, alpha=0.7, cmap="plasma")
    plt.colorbar(sc, label="Action")
    plt.xlabel("PC1(Δz)")
    plt.ylabel("PC2(Δz)")
    plt.title(f"{model_family}: Δz PCA (PC1 vs PC2) colored by action")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_delta_pca_pc1_pc2_action.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")


def plot_tsne(model_family, base_norm, aligned_norm, actions, fig_dir: Path, max_points=2000):
    """
    t-SNE of base vs aligned encodings.
    """
    n = base_norm.shape[0]
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        base_sub = base_norm[idx]
        aligned_sub = aligned_norm[idx]
        actions_sub = actions[idx]
    else:
        base_sub = base_norm
        aligned_sub = aligned_norm
        actions_sub = actions

    combined = np.vstack([base_sub, aligned_sub])

    # PCA to 50D for speed
    pca = PCA(n_components=min(50, combined.shape[1]))
    combined_pca = pca.fit_transform(combined)

    print(f"    [t-SNE] Running t-SNE on {combined_pca.shape[0]} points...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    embedded = tsne.fit_transform(combined_pca)

    base_emb = embedded[: base_sub.shape[0]]
    aligned_emb = embedded[base_sub.shape[0] :]

    # Base vs aligned
    plt.figure(figsize=(6, 4))
    plt.scatter(base_emb[:, 0], base_emb[:, 1], s=6, alpha=0.5, label="Base")
    plt.scatter(aligned_emb[:, 0], aligned_emb[:, 1], s=6, alpha=0.5, label="Aligned")
    plt.legend()
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"{model_family}: t-SNE Base vs Aligned")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_tsne_base_vs_aligned.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")

    # Base colored by action
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(base_emb[:, 0], base_emb[:, 1], c=actions_sub, s=6, alpha=0.7, cmap="viridis")
    plt.colorbar(sc, label="Action")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"{model_family}: t-SNE Base colored by Action")
    plt.tight_layout()
    out = fig_dir / f"{model_family}_tsne_base_colored_action.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"    [fig] {out}")


# -----------------------------------------------------------------------------
# TABLE HELPERS (PER MODEL)
# -----------------------------------------------------------------------------

def save_global_stats_csv(model_family, summary, out_dir: Path):
    out_path = out_dir / f"{model_family}_global_stats.csv"
    fields = [
        "model_family",
        "n_samples",
        "latent_dim",
        "w2",
        "mean_action",
        "std_action",
        "mean_geodesic_sq",
        "mean_action_over_w2",
        "corr_action_toxicity_r",
        "corr_action_toxicity_p",
        "corr_action_geod_r",
        "corr_action_geod_p",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        g = summary["global"]
        row = {
            "model_family": summary["model_family"],
            "n_samples": summary["n_samples"],
            "latent_dim": summary["latent_dim"],
            "w2": g["w2"],
            "mean_action": g["mean_action"],
            "std_action": g["std_action"],
            "mean_geodesic_sq": g["mean_geodesic_sq"],
            "mean_action_over_w2": g["mean_action_over_w2"],
            "corr_action_toxicity_r": g["corr_action_toxicity"]["r"],
            "corr_action_toxicity_p": g["corr_action_toxicity"]["p"],
            "corr_action_geod_r": g["corr_action_geodesic_sq"]["r"],
            "corr_action_geod_p": g["corr_action_geodesic_sq"]["p"],
        }
        writer.writerow(row)
    print(f"    [csv] {out_path}")


def save_tox_bins_csv(model_family, summary, out_dir: Path):
    out_path = out_dir / f"{model_family}_toxicity_bins.csv"
    fields = [
        "model_family",
        "bin_label",
        "count",
        "skipped",
        "toxicity_mean",
        "w2",
        "mean_action",
        "mean_geodesic_sq",
        "mean_action_over_w2",
        "mean_ratio_action_over_geod",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, stats in summary["toxicity_bins"].items():
            row = {
                "model_family": summary["model_family"],
                "bin_label": label,
                "count": stats.get("count", 0),
                "skipped": stats.get("skipped", False),
                "toxicity_mean": stats.get("toxicity_mean", np.nan),
                "w2": stats.get("w2", np.nan),
                "mean_action": stats.get("mean_action", np.nan),
                "mean_geodesic_sq": stats.get("mean_geodesic_sq", np.nan),
                "mean_action_over_w2": stats.get("mean_action_over_w2", np.nan),
                "mean_ratio_action_over_geod": stats.get("mean_ratio_action_over_geod", np.nan),
            }
            writer.writerow(row)
    print(f"    [csv] {out_path}")


# -----------------------------------------------------------------------------
# ANALYSIS PER MODEL
# -----------------------------------------------------------------------------

def analyze_model(args_tuple):
    """
    Wrapper for multiprocessing: (model_family, base_path, sae_subdir, ode_subdir, data_dir, output_dir)
    """
    (model_family, base_path, sae_subdir, ode_subdir, data_dir, output_dir) = args_tuple

    print("\n" + "=" * 80)
    print(f"[{model_family}] GEOMETRIC ANALYSIS START")
    print("=" * 80)

    base_path = Path(base_path).resolve()
    sae_dir = base_path / sae_subdir
    ode_dir = base_path / ode_subdir
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    ensure_dir(output_dir)

    # Per-model figure dir
    fig_dir = ensure_dir(output_dir / "figures" / model_family)

    # 1. Load encodings
    base_enc, aligned_enc = load_encodings(sae_dir, model_family)
    n_samples, d = base_enc.shape

    # 2. Load actions
    actions = load_actions(ode_dir, model_family, n_samples)

    # 3. Load toxicity
    toxicity = load_toxicity_scores(data_dir, n_samples)

    # 4. Normalize features (shared)
    base_norm, aligned_norm = normalize_features(base_enc, aligned_enc)

    # 5. Global OT and geodesic
    print(f"[{model_family}] Global OT & geodesic...")
    w2_global = compute_w2(base_norm, aligned_norm, reg=0.01)

    delta = aligned_norm - base_norm
    geodesic_sq = np.sum(delta**2, axis=1)  # per-sample straight-line cost
    ratio = actions / (geodesic_sq + 1e-8)

    print(f"[{model_family}] Mean action:         {actions.mean():.4f}")
    print(f"[{model_family}] Mean geod_sq:       {geodesic_sq.mean():.4f}")
    print(f"[{model_family}] W2_global:          {w2_global:.4f}")
    print(f"[{model_family}] mean(action)/W2:    {actions.mean()/w2_global:.4f}")

    # correlation of action with toxicity and geodesic_sq
    r_act_tox, p_act_tox = pearsonr(actions, toxicity)
    r_act_geo, p_act_geo = pearsonr(actions, geodesic_sq)
    print(f"[{model_family}] corr(Action, Toxicity): r={r_act_tox:.3f}, p={p_act_tox:.3e}")
    print(f"[{model_family}] corr(Action, Geod^2):  r={r_act_geo:.3f}, p={p_act_geo:.3e}")

    # 6. Toxicity-binned analysis
    print(f"[{model_family}] Toxicity-binned OT and curvature...")

    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    bin_labels = ["benign_0-0.25", "borderline_0.25-0.5",
                  "concerning_0.5-0.75", "toxic_0.75-1.0"]

    bin_stats = {}

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        label = bin_labels[i]
        mask = (toxicity >= lo) & (toxicity < hi)
        count = int(mask.sum())
        if count < 10:
            print(f"  [{model_family}] Skipping bin {label}: only {count} samples")
            bin_stats[label] = {
                "count": count,
                "skipped": True,
            }
            continue

        base_bin = base_norm[mask]
        aligned_bin = aligned_norm[mask]
        actions_bin = actions[mask]
        geod_bin = geodesic_sq[mask]
        ratio_bin = ratio[mask]
        tox_bin = toxicity[mask]

        print(f"  [{model_family}] Bin {label}: {count} samples")
        w2_bin = compute_w2(base_bin, aligned_bin, reg=0.01)

        bin_stats[label] = {
            "count": count,
            "toxicity_mean": float(tox_bin.mean()),
            "w2": float(w2_bin),
            "mean_action": float(actions_bin.mean()),
            "mean_geodesic_sq": float(geod_bin.mean()),
            "mean_action_over_w2": float(actions_bin.mean() / (w2_bin + 1e-8)),
            "mean_ratio_action_over_geod": float(ratio_bin.mean()),
        }

    # 7. Alignment subspace (PCA on Δz)
    print(f"[{model_family}] Alignment subspace (PCA on Δz)...")
    pca = PCA(n_components=3)
    delta_pca = pca.fit_transform(delta)  # (N, 3)
    explained = pca.explained_variance_ratio_.tolist()

    pc1 = delta_pca[:, 0]
    r_pc1_tox, p_pc1_tox = pearsonr(pc1, toxicity)
    r_pc1_act, p_pc1_act = pearsonr(pc1, actions)
    print(f"[{model_family}] PC1-Δz corr(Toxicity): r={r_pc1_tox:.3f}, p={p_pc1_tox:.3e}")
    print(f"[{model_family}] PC1-Δz corr(Action):   r={r_pc1_act:.3f}, p={p_pc1_act:.3e}")

    # 8. Summarize (structured JSON, no huge arrays)
    summary = {
        "model_family": model_family,
        "n_samples": int(n_samples),
        "latent_dim": int(d),
        "global": {
            "w2": float(w2_global),
            "mean_action": float(actions.mean()),
            "std_action": float(actions.std()),
            "mean_geodesic_sq": float(geodesic_sq.mean()),
            "mean_action_over_w2": float(actions.mean() / (w2_global + 1e-8)),
            "corr_action_toxicity": {
                "r": float(r_act_tox),
                "p": float(p_act_tox),
            },
            "corr_action_geodesic_sq": {
                "r": float(r_act_geo),
                "p": float(p_act_geo),
            },
        },
        "toxicity_bins": bin_stats,
        "alignment_subspace": {
            "explained_variance_ratio": explained,
            "pc1_corr_toxicity": {
                "r": float(r_pc1_tox),
                "p": float(p_pc1_tox),
            },
            "pc1_corr_action": {
                "r": float(r_pc1_act),
                "p": float(p_pc1_act),
            },
        },
    }

    # 9. Save JSON summary
    out_json = output_dir / f"{model_family}_geom_analysis.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{model_family}] Saved summary JSON to {out_json}")

    # 10. Save per-model CSV tables
    save_global_stats_csv(model_family, summary, output_dir)
    save_tox_bins_csv(model_family, summary, output_dir)

    # 11. Generate per-model plots
    print(f"[{model_family}] Generating figures...")
    plot_action_hist(model_family, actions, fig_dir)
    plot_action_vs_toxicity(model_family, actions, toxicity, fig_dir)
    plot_action_by_tox_bin(model_family, actions, toxicity, fig_dir)
    plot_ratio_hist(model_family, ratio, fig_dir)
    plot_tox_bin_w2_and_action(model_family, bin_stats, fig_dir)
    plot_pca_delta(model_family, delta_pca, toxicity, actions, explained, fig_dir)
    plot_tsne(model_family, base_norm, aligned_norm, actions, fig_dir)

    print(f"[{model_family}] GEOMETRIC ANALYSIS DONE\n")

    return summary


# -----------------------------------------------------------------------------
# CROSS-MODEL PLOTS & TABLES
# -----------------------------------------------------------------------------

def save_all_models_global_csv(results, out_dir: Path):
    out_path = out_dir / "all_models_global_stats.csv"
    fields = [
        "model_family",
        "n_samples",
        "latent_dim",
        "w2",
        "mean_action",
        "std_action",
        "mean_geodesic_sq",
        "mean_action_over_w2",
        "corr_action_toxicity_r",
        "corr_action_toxicity_p",
        "corr_action_geod_r",
        "corr_action_geod_p",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in results:
            g = summary["global"]
            row = {
                "model_family": summary["model_family"],
                "n_samples": summary["n_samples"],
                "latent_dim": summary["latent_dim"],
                "w2": g["w2"],
                "mean_action": g["mean_action"],
                "std_action": g["std_action"],
                "mean_geodesic_sq": g["mean_geodesic_sq"],
                "mean_action_over_w2": g["mean_action_over_w2"],
                "corr_action_toxicity_r": g["corr_action_toxicity"]["r"],
                "corr_action_toxicity_p": g["corr_action_toxicity"]["p"],
                "corr_action_geod_r": g["corr_action_geodesic_sq"]["r"],
                "corr_action_geod_p": g["corr_action_geodesic_sq"]["p"],
            }
            writer.writerow(row)
    print(f"[all_models] Saved global stats CSV to {out_path}")


def plot_cross_model_bars(results, out_dir: Path):
    fig_dir = ensure_dir(out_dir / "figures")
    models = [r["model_family"] for r in results]
    w2_vals = [r["global"]["w2"] for r in results]
    mean_actions = [r["global"]["mean_action"] for r in results]
    mean_ratio = [r["global"]["mean_action_over_w2"] for r in results]

    x = np.arange(len(models))

    # W2
    plt.figure(figsize=(6, 4))
    plt.bar(x, w2_vals)
    plt.xticks(x, models)
    plt.ylabel("Global W2")
    plt.title("Global Wasserstein-2 Distance by Model")
    plt.tight_layout()
    out = fig_dir / "all_models_global_w2.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[all_models][fig] {out}")

    # Mean action
    plt.figure(figsize=(6, 4))
    plt.bar(x, mean_actions)
    plt.xticks(x, models)
    plt.ylabel("Mean Action")
    plt.title("Mean Neural-ODE Action by Model")
    plt.tight_layout()
    out = fig_dir / "all_models_mean_action.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[all_models][fig] {out}")

    # Mean action / W2
    plt.figure(figsize=(6, 4))
    plt.bar(x, mean_ratio)
    plt.xticks(x, models)
    plt.ylabel("Mean(Action) / W2")
    plt.title("Curvature Ratio (Mean Action / W2) by Model")
    plt.tight_layout()
    out = fig_dir / "all_models_mean_action_over_w2.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[all_models][fig] {out}")


# -----------------------------------------------------------------------------
# MAIN (PARALLEL OVER MODELS)
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run OT + geometric alignment analysis for multiple models."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path for the project (e.g., /home/ark89/scratch_pi_ds256/ark89/Elicitation-Geometry).",
    )
    parser.add_argument(
        "--sae_subdir",
        type=str,
        default="new_approach/sae_output_ultra_backup_first_half",
        help="Relative subdirectory (from base_path) where SAE encodings live.",
    )
    parser.add_argument(
        "--ode_subdir",
        type=str,
        default="new_approach/ode_output_ultra",
        help="Relative subdirectory (from base_path) where Neural-ODE outputs live.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing prompts_split_*.json (absolute or relative to base_path).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="new_approach/analysis_output",
        help="Output directory (relative to base_path) for analysis JSONs, CSVs, and figures.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["llama2", "falcon", "mistral"],
        help="List of model families to analyze.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: len(models)).",
    )

    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    # data_dir may be absolute or relative to base_path
    data_dir = (
        Path(args.data_dir).resolve()
        if Path(args.data_dir).is_absolute()
        else (base_path / args.data_dir).resolve()
    )
    output_dir = (base_path / args.output_dir).resolve()
    ensure_dir(output_dir)

    models = args.models
    n_proc = args.processes or len(models)

    # Prepare args for each model
    per_model_args = [
        (
            model_family,
            str(base_path),
            args.sae_subdir,
            args.ode_subdir,
            str(data_dir),
            str(output_dir),
        )
        for model_family in models
    ]

    print("=" * 80)
    print("RUNNING GEOMETRIC ANALYSIS FOR MODELS:", ", ".join(models))
    print("=" * 80)

    # Parallel over models
    with Pool(processes=n_proc) as pool:
        results = pool.map(analyze_model, per_model_args)

    # Save a combined summary JSON
    combined = {res["model_family"]: res for res in results}
    combined_path = output_dir / "all_models_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[all_models] Combined summary JSON saved to {combined_path}")

    # Save combined CSV + cross-model plots
    save_all_models_global_csv(results, output_dir)
    plot_cross_model_bars(results, output_dir)

    print("\n" + "=" * 80)
    print("✅ ALL MODELS ANALYSIS COMPLETE")
    print(f"All outputs in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
