#!/usr/bin/env python3
"""
Analyze SAE outputs for base vs aligned models across llama2 / falcon / mistral.

Uses the structure:

new_sae_outputs/
  llama2/run_20251207_142926/...
  falcon/run_20251207_143425/...
  mistral/run_20251207_143155/...

and RealToxicityPrompts JSONs under:
  data/prompts_split_0.json ... data/prompts_split_5.json
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional: t-SNE / PCA
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not found; t-SNE/PCA plots will be skipped.")


# =============================================================================
# CONFIG ––– EDIT THIS IF YOU MOVE THINGS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
# Use the RealToxicityPrompts splits copied alongside the pooled numpy outputs.
# Any copy is fine as long as prompts_split_{0-5}.json are present and match the
# encoding order. Here we point to the llama2_base copy in numpy_activations.
DATA_DIR = PROJECT_ROOT / "data" / "numpy_activations" / "llama2_base"

@dataclass
class ModelConfig:
    family: str     # "llama2", "falcon", "mistral"
    prefix: str     # filename prefix
    run_dir: Path   # the specific SAE run directory


RUN_DIRS: Dict[str, ModelConfig] = {
    "llama2": ModelConfig(
        family="llama2",
        prefix="llama2",
        run_dir=PROJECT_ROOT / "new_sae_outputs" / "llama2" / "run_20251207_142926",
    ),
    "falcon": ModelConfig(
        family="falcon",
        prefix="falcon",
        run_dir=PROJECT_ROOT / "new_sae_outputs" / "falcon" / "run_20251207_143425",
    ),
    "mistral": ModelConfig(
        family="mistral",
        prefix="mistral",
        run_dir=PROJECT_ROOT / "new_sae_outputs" / "mistral" / "run_20251207_143155",
    ),
}

TOXICITY_BUCKETS = [
    (0.0, 0.25,  "benign"),
    (0.25, 0.5,  "borderline"),
    (0.5, 0.75,  "concerning"),
    (0.75, 1.01, "toxic"),
]

TOPK_FEATURES = 32
# Upper bound on samples to keep in RAM for analysis (helps avoid OOM)
MAX_ANALYSIS_SAMPLES = 1000
# t-SNE subsample size (applied after the global MAX_ANALYSIS_SAMPLES cap)
TSNE_SUBSAMPLE = 0  # set to 0 to skip t-SNE entirely


# =============================================================================
# UTILS
# =============================================================================

def ensure_dirs(run_dir: Path) -> Tuple[Path, Path]:
    fig = run_dir / "figures"
    tab = run_dir / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return fig, tab


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) if x.size > 1 else 0.0)


def bucket_indices(tox: np.ndarray, buckets=TOXICITY_BUCKETS):
    out = {}
    for lo, hi, label in buckets:
        mask = (tox >= lo) & (tox < hi)
        out[label] = np.where(mask)[0]
    return out


def load_toxicity_from_json(data_dir: Path, n_samples: int) -> np.ndarray:
    """
    Mirror of your geom script: read all prompts_split_*.json and
    take the first n_samples toxicity scores.
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
    if all_tox.shape[0] < n_samples:
        raise ValueError(
            f"Toxicity entries {all_tox.shape[0]} < encodings {n_samples}. "
            f"Check alignment between prompts and pooled activations."
        )

    subset = all_tox[:n_samples]
    return subset


# =============================================================================
# PER-MODEL ANALYSIS
# =============================================================================

def analyze_model(cfg: ModelConfig):
    print(f"\n==================== {cfg.family.upper()} ====================")
    run_dir = cfg.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    fig_dir, tab_dir = ensure_dirs(run_dir)
    pref = cfg.prefix

    # ---- Load core SAE outputs ----
    # Use memmap to avoid loading full 9000x16384 arrays into RAM at once
    mm_base_codes    = np.load(run_dir / f"{pref}_base_sae_encoded.npy", mmap_mode="r")
    mm_aligned_codes = np.load(run_dir / f"{pref}_aligned_sae_encoded.npy", mmap_mode="r")
    mm_base_err      = np.load(run_dir / f"{pref}_base_recon_error.npy", mmap_mode="r")
    mm_aligned_err   = np.load(run_dir / f"{pref}_aligned_recon_error.npy", mmap_mode="r")
    mm_delta_err     = np.load(run_dir / f"{pref}_delta_recon_error.npy", mmap_mode="r")
    base_mean        = load_array(run_dir / f"{pref}_base_feature_mean.npy").astype(np.float32)
    aligned_mean     = load_array(run_dir / f"{pref}_aligned_feature_mean.npy").astype(np.float32)
    mean_shift       = load_array(run_dir / f"{pref}_feature_mean_shift.npy").astype(np.float32)

    n_samples = mm_base_codes.shape[0]
    if n_samples > MAX_ANALYSIS_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, MAX_ANALYSIS_SAMPLES, replace=False)
        print(f"  Subsampled to {len(idx)} samples for analysis (cap={MAX_ANALYSIS_SAMPLES})")
    else:
        idx = np.arange(n_samples)

    # Materialize only the needed subset into RAM as float32
    base_codes    = np.array(mm_base_codes[idx], dtype=np.float32)
    aligned_codes = np.array(mm_aligned_codes[idx], dtype=np.float32)
    base_err      = np.array(mm_base_err[idx], dtype=np.float32)
    aligned_err   = np.array(mm_aligned_err[idx], dtype=np.float32)
    delta_err     = np.array(mm_delta_err[idx], dtype=np.float32)
    n_samples = base_codes.shape[0]

    print(f"  base_codes shape:    {base_codes.shape}")
    print(f"  aligned_codes shape: {aligned_codes.shape}")
    print(f"  base_err shape:      {base_err.shape}")
    print(f"  aligned_err shape:   {aligned_err.shape}")
    print(f"  mean_shift shape:    {mean_shift.shape}")

    assert aligned_codes.shape[0] == n_samples
    assert base_err.shape[0] == n_samples == aligned_err.shape[0] == delta_err.shape[0]

    # ---- Toxicity (from RealToxicityPrompts JSON) ----
    tox = load_toxicity_from_json(DATA_DIR, n_samples)
    print(f"  Loaded {tox.shape[0]} toxicity scores")

    # =============================================================================
    # A. OFF-MANIFOLDNESS (RECON ERRORS)
    # =============================================================================

    base_mean_err, base_std_err = mean_std(base_err)
    aligned_mean_err, aligned_std_err = mean_std(aligned_err)
    delta_mean_err, delta_std_err = mean_std(delta_err)

    print(f"  Overall base   recon error: {base_mean_err:.4f} ± {base_std_err:.4f}")
    print(f"  Overall aligned recon error: {aligned_mean_err:.4f} ± {aligned_std_err:.4f}")
    print(f"  Overall delta  recon error: {delta_mean_err:.4f} ± {delta_std_err:.4f}")

    overall_df = pd.DataFrame(
        {
            "model_family": [cfg.family],
            "base_mean_err": [base_mean_err],
            "base_std_err": [base_std_err],
            "aligned_mean_err": [aligned_mean_err],
            "aligned_std_err": [aligned_std_err],
            "delta_mean_err": [delta_mean_err],
            "delta_std_err": [delta_std_err],
        }
    )
    overall_df.to_csv(tab_dir / f"{pref}_overall_recon_error.csv", index=False)

    # ---- Recon error vs toxicity bucket ----
    bucket_idx = bucket_indices(tox)
    rows = []
    labels = []
    base_means, base_stds = [], []
    aligned_means, aligned_stds = [], []
    delta_means, delta_stds = [], []

    for label, idx in bucket_idx.items():
        if idx.size == 0:
            continue

        b = base_err[idx]
        a = aligned_err[idx]
        d = delta_err[idx]

        bm, bs = mean_std(b)
        am, as_ = mean_std(a)
        dm, ds = mean_std(d)

        rows.append(
            {
                "bucket": label,
                "n": idx.size,
                "base_mean_err": bm,
                "base_std_err": bs,
                "aligned_mean_err": am,
                "aligned_std_err": as_,
                "delta_mean_err": dm,
                "delta_std_err": ds,
            }
        )

        labels.append(label)
        base_means.append(bm)
        base_stds.append(bs)
        aligned_means.append(am)
        aligned_stds.append(as_)
        delta_means.append(dm)
        delta_stds.append(ds)

    bucket_df = pd.DataFrame(rows)
    bucket_df.to_csv(tab_dir / f"{pref}_recon_error_by_toxicity_bucket.csv", index=False)

    x = np.arange(len(labels))
    width = 0.27

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, base_means, width, yerr=base_stds, capsize=4, label="Base")
    plt.bar(x,        aligned_means, width, yerr=aligned_stds, capsize=4, label="Aligned")
    plt.bar(x + width, delta_means, width, yerr=delta_stds, capsize=4, label="Delta")
    plt.xticks(x, labels)
    plt.ylabel("Reconstruction error (L2)")
    plt.title(f"{cfg.family}: reconstruction error by toxicity bucket")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{pref}_recon_error_by_toxicity_bucket.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label, idx in bucket_idx.items():
        if idx.size == 0:
            continue
        plt.hist(
            delta_err[idx],
            bins=50,
            alpha=0.5,
            label=f"{label} (n={idx.size})",
            density=True,
        )
    plt.xlabel("Delta reconstruction error (aligned - base)")
    plt.ylabel("Density")
    plt.title(f"{cfg.family}: Δ reconstruction error by toxicity bucket")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{pref}_delta_recon_error_by_toxicity_hist.png", dpi=200)
    plt.close()

    # =============================================================================
    # B. FEATURE-LEVEL STATS (SAE LATENT GEOMETRY)
    # =============================================================================

    mean_shift_norm = float(np.linalg.norm(mean_shift))
    print(f"  ‖feature_mean_shift‖₂ = {mean_shift_norm:.4f}")
    with open(tab_dir / f"{pref}_feature_mean_shift_norm.json", "w") as f:
        json.dump({"mean_shift_norm": mean_shift_norm}, f, indent=2)

    abs_shift = np.abs(mean_shift)
    top_idx = np.argsort(-abs_shift)[:TOPK_FEATURES]
    top_vals = mean_shift[top_idx]

    feature_df = pd.DataFrame(
        {
            "feature_index": top_idx,
            "mean_shift": top_vals,
            "abs_mean_shift": np.abs(top_vals),
            "base_feature_mean": base_mean[top_idx],
            "aligned_feature_mean": aligned_mean[top_idx],
        }
    )
    feature_df.to_csv(tab_dir / f"{pref}_top_{TOPK_FEATURES}_shifted_features.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(top_idx)), top_vals)
    plt.xlabel("Top-shifted SAE feature rank")
    plt.ylabel("Mean shift (aligned - base)")
    plt.title(f"{cfg.family}: top-{TOPK_FEATURES} shifted SAE features")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{pref}_top_{TOPK_FEATURES}_shifted_features.png", dpi=200)
    plt.close()

    # =============================================================================
    # C. OPTIONAL: Latent Geometry Visualization
    # =============================================================================

    if not HAS_SKLEARN or TSNE_SUBSAMPLE <= 0:
        return

    n_total = base_codes.shape[0]
    if n_total > TSNE_SUBSAMPLE:
        idx = np.random.choice(n_total, TSNE_SUBSAMPLE, replace=False)
    else:
        idx = np.arange(n_total)

    base_sub = base_codes[idx]
    aligned_sub = aligned_codes[idx]
    n_sub = base_sub.shape[0]

    X = np.vstack([base_sub, aligned_sub])
    labels_ma = np.array(["base"] * n_sub + ["aligned"] * n_sub)
    tox_sub = tox[idx]

    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)
    X_2d = tsne.fit_transform(X_pca)

    mask_base = (labels_ma == "base")
    mask_aligned = (labels_ma == "aligned")

    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[mask_base, 0], X_2d[mask_base, 1], s=5, alpha=0.4, label="Base")
    plt.scatter(X_2d[mask_aligned, 0], X_2d[mask_aligned, 1], s=5, alpha=0.4, label="Aligned")
    plt.title(f"{cfg.family}: SAE latent t-SNE (base vs aligned)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{pref}_tsne_base_vs_aligned.png", dpi=250)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(X_2d[:n_sub, 0], X_2d[:n_sub, 1],
                     c=tox_sub, s=5, alpha=0.5, cmap="viridis")
    plt.colorbar(sc, label="toxicity")
    plt.title(f"{cfg.family}: SAE latent t-SNE (base colored by toxicity)")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{pref}_tsne_base_toxicity.png", dpi=250)
    plt.close()


def main():
    for name, cfg in RUN_DIRS.items():
        analyze_model(cfg)

    # Global summary of mean-shift norms
    rows = []
    for name, cfg in RUN_DIRS.items():
        tab_dir = cfg.run_dir / "tables"
        path = tab_dir / f"{cfg.prefix}_feature_mean_shift_norm.json"
        if not path.exists():
            continue
        with open(path, "r") as f:
            data = json.load(f)
        rows.append(
            {"model_family": cfg.family, "mean_shift_norm": data["mean_shift_norm"]}
        )

    if rows:
        out_dir = PROJECT_ROOT / "new_sae_outputs"
        out_dir.mkdir(exist_ok=True)
        df = pd.DataFrame(rows)
        out_path = out_dir / "feature_mean_shift_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"\nWrote global feature-mean-shift summary to: {out_path}")


if __name__ == "__main__":
    main()
