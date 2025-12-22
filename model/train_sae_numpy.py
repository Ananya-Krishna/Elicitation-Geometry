#!/usr/bin/env python3
"""
SAE training on pooled NumPy activations (base-only) with:
  - A: off-manifoldness via reconstruction errors for base vs aligned
  - B: feature-level statistics (per-feature means and shifts)
  - Basic timing measurements for training and encoding

Assumes upstream data_curation_numpy_parallel_data.py produced:
    numpy_activations/
        llama2_base/
            llama2_base_split_0_pooled.npy
            ...
        llama2_aligned/
            llama2_aligned_split_0_pooled.npy
            ...
        falcon_base/...
        falcon_aligned/...
        mistral_base/...
        mistral_aligned/...
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ============================================================================
# SPARSE AUTOENCODER MODEL
# ============================================================================

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for compressing LLM activations"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16384,
        sparsity_coef: float = 1e-3,  # reserved for future use
        l1_coef: float = 1e-4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef
        self.l1_coef = l1_coef

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Overcomplete, sparse codes with ReLU nonlinearity
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """MSE reconstruction + L1 on codes, plus sparsity metric."""
        recon_loss = nn.functional.mse_loss(x_recon, x)
        l1_loss = torch.mean(torch.abs(z))
        total_loss = recon_loss + self.l1_coef * l1_loss
        sparsity = (torch.abs(z) < 0.01).float().mean()
        return {
            "total": total_loss,
            "recon": recon_loss,
            "l1": l1_loss,
            "sparsity": sparsity,
        }


# ============================================================================
# DATASET: NUMPY-POOLED ACTIVATIONS
# ============================================================================

class UltraFastActivationDataset(Dataset):
    """
    Uses pre-pooled NumPy activations for a given MODEL_KEY.

    Expects files like:
        {data_dir}/{model_name}_split_{k}_pooled.npy

    Typical layout:

        data_root/
          llama2_base/
            llama2_base_split_0_pooled.npy
            ...
          llama2_aligned/
            llama2_aligned_split_0_pooled.npy
            ...

    - Each .npy is shape (num_prompts_in_split, hidden_dim)
    - We optionally subsample and then preload selected rows into RAM.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        max_samples: Optional[int] = None,
        subsample: float = 1.0,
    ):
        """
        Args:
            data_dir: Directory containing pooled activations for this model.
            model_name: e.g., "llama2_base".
            max_samples: Optional cap on number of samples.
            subsample: Fraction of data to use (0.0, 1.0].
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name

        # ---- File discovery: look for *_pooled.npy in this directory ----
        split_files = sorted(
            self.data_dir.glob(f"{model_name}_split_*_pooled_fixed.npy")
        )

        # Fallback: recursive search in case of nested structure
        if not split_files:
            split_files = sorted(
                self.data_dir.glob(f"**/{model_name}_split_*_pooled_fixed.npy")
            )

        if not split_files:
            raise ValueError(
                f"No pooled .npy files found for {model_name} in {data_dir}. "
                f"Expected pattern like {model_name}_split_*_pooled_fixed.npy"
            )

        print(f"Found {len(split_files)} pooled split files for {model_name}")
        self.split_files = split_files

        # ---- Open arrays (mmap) and record sizes ----
        print("Opening pooled .npy files (mmap) for fast access...")
        self.arrays = [np.load(f, mmap_mode="r") for f in split_files]
        self.split_sizes = [arr.shape[0] for arr in self.arrays]

        # Sanity check: dimensions
        hidden_dims = {arr.shape[1] for arr in self.arrays}
        if len(hidden_dims) != 1:
            raise ValueError(f"Inconsistent hidden_dim across splits: {hidden_dims}")
        self.hidden_dim = hidden_dims.pop()
        self.total_available = int(sum(self.split_sizes))

        # ---- Determine how many samples to actually use ----
        if not (0.0 < subsample <= 1.0):
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        effective_total = int(self.total_available * subsample)
        if max_samples is not None:
            effective_total = min(effective_total, max_samples)
        effective_total = max(1, effective_total)

        self.num_samples = effective_total
        print(
            f"Total available samples: {self.total_available:,} "
            f"→ using {self.num_samples:,} samples "
            f"({self.num_samples / self.total_available * 100:.1f}%)"
        )

        # ---- Build (split_idx, local_idx) indices for the first effective_total samples ----
        self.indices = []
        remaining = self.num_samples
        for split_idx, n in enumerate(self.split_sizes):
            take = min(n, remaining)
            for local_idx in range(take):
                self.indices.append((split_idx, local_idx))
            remaining -= take
            if remaining <= 0:
                break

        # ---- Preload all selected rows into memory ----
        print(
            f"Pre-loading {len(self.indices):,} pooled samples into memory "
            f"(hidden_dim={self.hidden_dim})..."
        )
        data_list = []
        for split_idx, local_idx in tqdm(self.indices, desc="Loading pooled data"):
            vec = self.arrays[split_idx][local_idx]  # shape (hidden_dim,)
            data_list.append(vec.astype(np.float32))

        self.data = np.stack(data_list, axis=0)
        print(
            f"✓ Pre-loaded {self.data.shape[0]:,} samples "
            f"({self.data.nbytes / 1e6:.1f} MB)"
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])

    def close(self):
        """Release references; mmap arrays will be cleaned up when GC’d."""
        self.arrays = []
        self.data = None


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_sae(
    model: SparseAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_dir: Path,
) -> Dict[str, Any]:
    """Train the Sparse Autoencoder with history + best checkpoints."""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_l1": [],
        "val_l1": [],
        "train_sparsity": [],
        "val_sparsity": [],
        "epoch_seconds": [],
    }

    checkpoint = None

    for epoch in range(epochs):
        epoch_start = time.time()

        # --------------------- TRAIN ---------------------
        model.train()
        train_metrics = {"total": 0.0, "recon": 0.0, "l1": 0.0, "sparsity": 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for x in pbar:
            x = x.to(device, non_blocking=True)

            x_recon, z = model(x)
            losses = model.compute_loss(x, x_recon, z)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            for key in train_metrics:
                train_metrics[key] += losses[key].item()

            pbar.set_postfix(
                {"loss": losses["total"].item(), "sparsity": losses["sparsity"].item()}
            )

        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        # --------------------- VALIDATION ---------------------
        model.eval()
        val_metrics = {"total": 0.0, "recon": 0.0, "l1": 0.0, "sparsity": 0.0}
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for x in pbar:
                x = x.to(device, non_blocking=True)
                x_recon, z = model(x)
                losses = model.compute_loss(x, x_recon, z)
                for key in val_metrics:
                    val_metrics[key] += losses[key].item()
                pbar.set_postfix({"loss": losses["total"].item()})

        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        scheduler.step(val_metrics["total"])

        epoch_seconds = time.time() - epoch_start
        history["epoch_seconds"].append(epoch_seconds)

        print(f"\nEpoch {epoch + 1}/{epochs} (t = {epoch_seconds:.2f} s):")
        print(
            f"  Train - Loss: {train_metrics['total']:.6e}, "
            f"Recon: {train_metrics['recon']:.6e}, "
            f"L1: {train_metrics['l1']:.6e}, "
            f"Sparsity: {train_metrics['sparsity']:.2%}"
        )
        print(
            f"  Val   - Loss: {val_metrics['total']:.6e}, "
            f"Recon: {val_metrics['recon']:.6e}, "
            f"L1: {val_metrics['l1']:.6e}, "
            f"Sparsity: {val_metrics['sparsity']:.2%}"
        )

        # Record history
        history["train_loss"].append(train_metrics["total"])
        history["val_loss"].append(val_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["val_recon"].append(val_metrics["recon"])
        history["train_l1"].append(train_metrics["l1"])
        history["val_l1"].append(val_metrics["l1"])
        history["train_sparsity"].append(train_metrics["sparsity"])
        history["val_sparsity"].append(val_metrics["sparsity"])

        # Save best model
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["total"],
                "config": {
                    "input_dim": model.input_dim,
                    "hidden_dim": model.hidden_dim,
                    "l1_coef": model.l1_coef,
                    "sparsity_coef": model.sparsity_coef,
                },
            }
            torch.save(checkpoint, save_dir / "best_sae.pt")
            print(f"  ✓ Saved best model (val_loss: {val_metrics['total']:.6e})")

        # Periodic checkpoints every 5 epochs (if we have a checkpoint already)
        if checkpoint is not None and (epoch + 1) % 5 == 0:
            torch.save(checkpoint, save_dir / f"sae_epoch_{epoch + 1}.pt")

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ============================================================================
# ENCODING SCRIPT (with off-manifoldness + feature stats + timing)
# ============================================================================

def encode_and_save(
    sae: SparseAutoencoder,
    data_root: Path,
    model_name: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 256,
    subsample: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Encode activations using trained SAE and save to .npy.

    Returns:
        encoded_features: (N, hidden_dim)
        recon_errors: (N,) per-sample MSE
        feature_means: (hidden_dim,)
        elapsed_seconds: float
    """
    data_dir = data_root / model_name

    print(f"\nEncoding {model_name} activations...")
    print(f"  Data dir: {data_dir}")
    print(f"  Using subsample = {subsample * 100:.1f}%")

    dataset = UltraFastActivationDataset(
        data_dir=str(data_dir),
        model_name=model_name,
        max_samples=None,
        subsample=subsample,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    sae.eval()
    encoded_chunks = []
    err_chunks = []

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {model_name}"):
            batch = batch.to(device, non_blocking=True)
            x_recon, z = sae(batch)

            # Codes
            encoded_chunks.append(z.cpu().numpy())

            # Per-sample reconstruction error (MSE over features)
            err = torch.mean((x_recon - batch) ** 2, dim=1)
            err_chunks.append(err.cpu().numpy())

    elapsed = time.time() - start_time

    encoded_features = np.vstack(encoded_chunks)
    recon_errors = np.concatenate(err_chunks)
    feature_means = encoded_features.mean(axis=0)

    print(f"  Encoded shape: {encoded_features.shape}")
    print(f"  Recon error: mean={recon_errors.mean():.4e}, std={recon_errors.std():.4e}")
    print(f"  Sparsity (|z|<0.01): {(np.abs(encoded_features) < 0.01).mean():.2%}")
    print(f"  Encoding time: {elapsed:.2f} s")

    # Save arrays
    encoded_path = output_dir / f"{model_name}_sae_encoded.npy"
    recon_err_path = output_dir / f"{model_name}_recon_error.npy"
    feat_mean_path = output_dir / f"{model_name}_feature_mean.npy"

    np.save(encoded_path, encoded_features)
    np.save(recon_err_path, recon_errors)
    np.save(feat_mean_path, feature_means)

    print(f"  Saved encoded features to: {encoded_path}")
    print(f"  Saved recon errors to:     {recon_err_path}")
    print(f"  Saved feature means to:    {feat_mean_path}")

    dataset.close()
    return encoded_features, recon_errors, feature_means, elapsed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAE Training on NumPy Pooled Activations (base-only) "
                    "with off-manifoldness + feature stats + timing."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./numpy_activations",
        help="Root directory containing pooled NumPy activations",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="llama2",
        choices=["llama2", "falcon", "mistral"],
        help="Model family (base/aligned will be derived from this)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./sae_output_numpy",
        help="Root output directory for SAE runs",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name (default: timestamp-based, so no overwrites)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=16384, help="SAE hidden dimension"
    )
    parser.add_argument(
        "--l1_coef", type=float, default=1e-4, help="L1 regularization coefficient"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of BASE data to use for training (0.0–1.0]",
    )
    parser.add_argument(
        "--encode_subsample",
        type=float,
        default=1.0,
        help="Fraction of data to use for encoding for base/aligned",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build per-run output directory so runs don't overwrite each other
    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    else:
        run_name = args.run_name

    output_root = Path(args.output_root)
    run_dir = output_root / args.model_family / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print(f"SAE Training on NumPy Pooled Activations (BASE ONLY) - {args.model_family}")
    print(f"  Run directory:    {run_dir}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Train subsample:  {args.subsample * 100:.1f}% (base only)")
    print(f"  Encode subsample: {args.encode_subsample * 100:.1f}%")
    print("=" * 80 + "\n")

    # Base / aligned model names
    base_name = f"{args.model_family}_base"
    aligned_name = f"{args.model_family}_aligned"

    data_root = Path(args.data_root)

    # --------------------- LOAD BASE DATASET FOR TRAINING ---------------------
    base_data_dir = data_root / base_name

    base_dataset_full = UltraFastActivationDataset(
        data_dir=str(base_data_dir),
        model_name=base_name,
        max_samples=None,
        subsample=args.subsample,
    )

    # Split train/val (80/20)
    train_size = int(0.8 * len(base_dataset_full))
    val_size = len(base_dataset_full) - train_size
    train_subset, val_subset = random_split(base_dataset_full, [train_size, val_size])

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    print(f"Train samples (base): {len(train_subset):,}")
    print(f"Val samples (base):   {len(val_subset):,}")

    # Get input dimension from dataset
    sample = base_dataset_full[0]
    input_dim = sample.shape[0]
    print(f"Input dimension (hidden_dim): {input_dim}")

    # --------------------- INITIALIZE SAE ---------------------
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        l1_coef=args.l1_coef,
    ).to(device)

    print("\nSAE Architecture:")
    print(f"  Input:  {input_dim}")
    print(f"  Hidden: {args.hidden_dim}")
    print(f"  Overcompleteness ratio: {args.hidden_dim / input_dim:.2f}x")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # --------------------- TRAIN ---------------------
    print("\n" + "=" * 80)
    print("TRAINING SAE (BASE ONLY)")
    print("=" * 80 + "\n")

    train_start = time.time()
    history = train_sae(
        model=sae,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=run_dir,
    )
    train_elapsed = time.time() - train_start
    print(f"\nTotal training time: {train_elapsed:.2f} s "
          f"(avg per epoch: {train_elapsed / args.epochs:.2f} s)")

    # We can free the full base dataset's underlying arrays if desired
    base_dataset_full.close()

    # --------------------- LOAD BEST MODEL ---------------------
    checkpoint_path = run_dir / "best_sae.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sae.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")

    # --------------------- ENCODING ---------------------
    print("\n" + "=" * 80)
    print("ENCODING ACTIVATIONS (BASE + ALIGNED)")
    print("=" * 80 + "\n")
    print(
        f"Encoding only {args.encode_subsample * 100:.1f}% of data for each model "
        f"(set --encode_subsample 1.0 for full encoding).\n"
    )

    # Encode base
    base_encoded, base_recon_err, base_feat_mean, base_enc_secs = encode_and_save(
        sae=sae,
        data_root=data_root,
        model_name=base_name,
        output_dir=run_dir,
        device=device,
        batch_size=args.batch_size,
        subsample=args.encode_subsample,
    )

    # Encode aligned
    aligned_encoded, aligned_recon_err, aligned_feat_mean, aligned_enc_secs = encode_and_save(
        sae=sae,
        data_root=data_root,
        model_name=aligned_name,
        output_dir=run_dir,
        device=device,
        batch_size=args.batch_size,
        subsample=args.encode_subsample,
    )

    # --------------------- OFF-MANIFOLDNESS (A) & FEATURE SHIFTS (B) ---------------------
    summary: Dict[str, Any] = {
        "training_seconds": float(train_elapsed),
        "training_seconds_per_epoch": float(train_elapsed / args.epochs),
        "base_encoding_seconds": float(base_enc_secs),
        "aligned_encoding_seconds": float(aligned_enc_secs),
        "base_mean_recon_error": float(base_recon_err.mean()),
        "aligned_mean_recon_error": float(aligned_recon_err.mean()),
        "base_recon_error_std": float(base_recon_err.std()),
        "aligned_recon_error_std": float(aligned_recon_err.std()),
    }

    # Feature-level shifts (B)
    if base_feat_mean.shape != aligned_feat_mean.shape:
        print("⚠️ Feature mean shapes differ, cannot compute feature-level shift.")
    else:
        feat_shift = aligned_feat_mean - base_feat_mean
        feat_shift_path = run_dir / f"{args.model_family}_feature_mean_shift.npy"
        np.save(feat_shift_path, feat_shift)
        print(f"\nSaved feature mean shift (aligned - base) to: {feat_shift_path}")

        summary["feature_mean_shift_abs_mean"] = float(np.abs(feat_shift).mean())
        summary["feature_mean_shift_abs_max"] = float(np.abs(feat_shift).max())

    # Off-manifoldness Δerr (A)
    if base_recon_err.shape == aligned_recon_err.shape:
        delta_err = aligned_recon_err - base_recon_err
        delta_err_path = run_dir / f"{args.model_family}_delta_recon_error.npy"
        np.save(delta_err_path, delta_err)
        print(f"Saved delta recon error (aligned - base) to: {delta_err_path}")

        summary["delta_recon_error_mean"] = float(delta_err.mean())
        summary["delta_recon_error_std"] = float(delta_err.std())
        summary["delta_recon_error_abs_max"] = float(np.abs(delta_err).max())
    else:
        print("⚠️ base/aligned recon error shapes differ, cannot compute delta_err.")

    # Save timing + scalar metrics summary
    summary_path = run_dir / "timing_and_metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved timing + scalar metrics summary to: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ SAE TRAINING + ENCODING COMPLETE (BASE-ONLY SAE)")
    print("=" * 80)
    print(f"\nOutputs saved to: {run_dir}")
    print("  - best_sae.pt: Best model checkpoint")
    print("  - training_history.json: Training metrics (incl. epoch_seconds)")
    print(f"  - {base_name}_sae_encoded.npy:    Encoded base activations ({base_encoded.shape})")
    print(f"  - {aligned_name}_sae_encoded.npy: Encoded aligned activations ({aligned_encoded.shape})")
    print(f"  - {base_name}_recon_error.npy, {aligned_name}_recon_error.npy: per-sample recon errors")
    print("  - *_feature_mean.npy and feature_mean_shift.npy: feature-level stats")
    print("  - delta_recon_error.npy: off-manifoldness metric")
    print("  - timing_and_metrics_summary.json: training & encoding timings + scalar metrics")
    print(f"\nNext step:")
    print(f"  python train_neural_ode.py --model_family {args.model_family} "
          f"--sae_run {run_dir}")


if __name__ == "__main__":
    main()
