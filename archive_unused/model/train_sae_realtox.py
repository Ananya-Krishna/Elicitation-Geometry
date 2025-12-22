# model/train_sawe_realtox.py

"""
STEP 1: Train Sparse Autoencoder on Base Model Activations (RealToxicityPrompts)
NUMPY-ONLY VERSION (no HDF5 in training/encoding)

This script:
1. Loads pooled base/aligned activations from NumPy files:
      numpy_dir/{model_family}_base_pooled.npy
      numpy_dir/{model_family}_aligned_pooled.npy
2. Trains a sparse autoencoder [hidden_dim] → [sae_hidden_dim] (overcomplete, sparse)
3. Saves:
      best_sae.pt
      training_history.json
      sae_recon_error_hist.png
4. Encodes a shared subset of rows for both base and aligned:
      {model_family}_base_sae_encoded.npy
      {model_family}_aligned_sae_encoded.npy
      {model_family}_encoded_indices.npy

Usage:
    python model/train_sawe_realtox.py \
        --numpy_dir ./numpy_data \
        --model_family llama2 \
        --epochs 5 \
        --batch_size 512 \
        --subsample 0.5 \
        --encode_subsample 0.5 \
        --output_dir ./sae_output_ultra
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
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
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor):
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
# TRAINING LOOP
# ============================================================================

def train_sae(model, train_loader, val_loader, device, epochs, lr, save_dir: Path):
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
    }

    checkpoint = None

    for epoch in range(epochs):
        # --------------------- TRAIN ---------------------
        model.train()
        train_metrics = {"total": 0.0, "recon": 0.0, "l1": 0.0, "sparsity": 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for (x,) in pbar:
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
            for (x,) in pbar:
                x = x.to(device, non_blocking=True)
                x_recon, z = model(x)
                losses = model.compute_loss(x, x_recon, z)
                for key in val_metrics:
                    val_metrics[key] += losses[key].item()
                pbar.set_postfix({"loss": losses["total"].item()})

        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        scheduler.step(val_metrics["total"])

        print(f"\nEpoch {epoch + 1}/{epochs}:")
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

        # Periodic checkpoints every 5 epochs (optional)
        if checkpoint is not None and (epoch + 1) % 5 == 0:
            torch.save(checkpoint, save_dir / f"sae_epoch_{epoch + 1}.pt")

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ============================================================================
# SAE EVALUATION: RECONSTRUCTION ERROR HISTOGRAM
# ============================================================================

def evaluate_sae_reconstruction(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    device,
    batch_size: int,
    save_dir: Path,
):
    """
    Compute per-sample reconstruction error on the full base activation set,
    print summary stats, and save a histogram.
    """
    print("\nEvaluating SAE reconstruction on full base activations...")

    sae.eval()
    dataset = TensorDataset(torch.from_numpy(activations).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_losses = []

    with torch.no_grad():
        for (x,) in tqdm(loader, desc="Recon eval"):
            x = x.to(device, non_blocking=True)
            x_recon, z = sae(x)
            mse = ((x_recon - x) ** 2).mean(dim=1)  # per-sample MSE
            all_losses.append(mse.cpu().numpy())

    all_losses = np.concatenate(all_losses, axis=0)
    print(f"  Recon MSE: mean={all_losses.mean():.6e}, std={all_losses.std():.6e}")
    print(f"  Min={all_losses.min():.6e}, Max={all_losses.max():.6e}")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(all_losses, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Per-sample reconstruction MSE")
    plt.ylabel("Count")
    plt.title("SAE Reconstruction Error Distribution (Base Activations)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = save_dir / "sae_recon_error_hist.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved reconstruction error histogram to: {out_path}")


# ============================================================================
# ENCODING SCRIPT
# ============================================================================

def encode_and_save(
    sae: SparseAutoencoder,
    numpy_dir: Path,
    model_name: str,
    output_dir: Path,
    device,
    batch_size: int,
    encoded_indices: np.ndarray,
):
    """
    Encode pooled activations using trained SAE and save to .npy.

    Uses a shared encoded_indices array to ensure base/aligned are perfectly aligned.
    """
    print(f"\nEncoding pooled activations for {model_name}...")
    acts_file = numpy_dir / f"{model_name}_pooled.npy"
    if not acts_file.exists():
        raise FileNotFoundError(f"Pooled activations not found: {acts_file}")

    activations = np.load(acts_file)
    n_total = activations.shape[0]

    if encoded_indices.max() >= n_total:
        raise ValueError(
            f"encoded_indices max {encoded_indices.max()} exceeds "
            f"pooled activations size {n_total} for {model_name}"
        )

    subset = activations[encoded_indices]  # [K, D]
    dataset = TensorDataset(torch.from_numpy(subset).float())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    sae.eval()
    encoded_chunks = []

    with torch.no_grad():
        for (batch,) in tqdm(loader, desc=f"Encoding {model_name}"):
            batch = batch.to(device, non_blocking=True)
            z = sae.encode(batch)
            encoded_chunks.append(z.cpu().numpy())

    encoded_features = np.vstack(encoded_chunks)
    sparsity = (np.abs(encoded_features) < 0.01).mean()

    print(f"  Encoded shape: {encoded_features.shape}")
    print(f"  Sparsity: {sparsity:.2%}")

    output_file = output_dir / f"{model_name}_sae_encoded.npy"
    np.save(output_file, encoded_features)
    print(f"  Saved encoded features to: {output_file}")

    return encoded_features


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAE Training on pooled NumPy activations (RealToxicityPrompts)"
    )
    parser.add_argument(
        "--numpy_dir",
        type=str,
        default="./numpy_data",
        help="Directory with *_pooled.npy from convert_h5_to_numpy_realtox.py",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="llama2",
        choices=["llama2", "falcon", "mistral"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sae_output_ultra",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs (reduced for speed)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size (increased for speed)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
        help="Learning rate (increased for faster convergence)",
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
        default=0.5,
        help="Fraction of data for SAE training (0.5 = 50%)",
    )
    parser.add_argument(
        "--encode_subsample",
        type=float,
        default=0.5,
        help=(
            "Fraction of pooled data to encode & use downstream "
            "(0.5 = 50%%; use 1.0 for full dataset)"
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    numpy_dir = Path(args.numpy_dir)
    train_model = f"{args.model_family}_base"

    print("\n" + "=" * 80)
    print(f"SAE TRAINING on pooled NumPy activations: {train_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Subsample (train): {args.subsample * 100:.1f}%")
    print("=" * 80 + "\n")

    # --------------------- LOAD POOLED BASE ACTIVATIONS ---------------------
    base_file = numpy_dir / f"{train_model}_pooled.npy"
    if not base_file.exists():
        raise FileNotFoundError(
            f"Pooled activations for base model not found: {base_file}. "
            f"Run convert_h5_to_numpy_realtox.py first."
        )
    base_acts = np.load(base_file).astype(np.float32)
    n_samples, input_dim = base_acts.shape

    print(f"Base pooled activations: {base_acts.shape}")
    print(f"Input dimension: {input_dim}")

    # Optional subsample for SAE training (but we still evaluate/encode on full dataset)
    if not (0.0 < args.subsample <= 1.0):
        raise ValueError(f"subsample must be in (0, 1], got {args.subsample}")
    n_train_total = int(n_samples * args.subsample)
    n_train_total = max(1, n_train_total)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_indices = indices[:n_train_total]

    train_tensor = torch.from_numpy(base_acts[train_indices]).float()
    dataset = TensorDataset(train_tensor)

    # Split train/val (80/20 of subsampled set)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

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

    print(f"Train samples (subsampled): {len(train_subset):,}")
    print(f"Val samples (subsampled):   {len(val_subset):,}")

    # --------------------- INITIALIZE SAE ---------------------
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        l1_coef=args.l1_coef,
    ).to(device)

    print("\nSAE Architecture:")
    print(f"  Input:  {input_dim}")
    print(f"  Hidden: {args.hidden_dim}")
    print(f"  Compression ratio: {args.hidden_dim / input_dim:.2f}x (overcomplete)")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # --------------------- TRAIN ---------------------
    print("\n" + "=" * 80)
    print("TRAINING SAE")
    print("=" * 80 + "\n")

    history = train_sae(
        model=sae,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=output_dir,
    )

    # --------------------- LOAD BEST MODEL ---------------------
    checkpoint = torch.load(output_dir / "best_sae.pt", map_location=device)
    sae.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best SAE model from epoch {checkpoint['epoch'] + 1}")

    # --------------------- EVALUATE RECONSTRUCTION ---------------------
    evaluate_sae_reconstruction(
        sae=sae,
        activations=base_acts,
        device=device,
        batch_size=args.batch_size,
        save_dir=output_dir,
    )

    # --------------------- ENCODING FOR BASE & ALIGNED ---------------------
    print("\n" + "=" * 80)
    print("ENCODING ACTIVATIONS FOR BASE & ALIGNED")
    print("=" * 80 + "\n")

    if not (0.0 < args.encode_subsample <= 1.0):
        raise ValueError(
            f"encode_subsample must be in (0, 1], got {args.encode_subsample}"
        )

    n_encode = int(n_samples * args.encode_subsample)
    n_encode = max(1, n_encode)
    rng = np.random.default_rng(seed=0)
    encoded_indices = rng.choice(n_samples, size=n_encode, replace=False)
    encoded_indices = np.sort(encoded_indices.astype(np.int64))

    indices_file = output_dir / f"{args.model_family}_encoded_indices.npy"
    np.save(indices_file, encoded_indices)
    print(
        f"Encoding subset size: {n_encode} / {n_samples} "
        f"({n_encode / n_samples * 100:.1f}%)"
    )
    print(f"Saved encoded_indices to: {indices_file}")

    base_name = f"{args.model_family}_base"
    aligned_name = f"{args.model_family}_aligned"

    base_encoded = encode_and_save(
        sae=sae,
        numpy_dir=numpy_dir,
        model_name=base_name,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        encoded_indices=encoded_indices,
    )

    aligned_encoded = encode_and_save(
        sae=sae,
        numpy_dir=numpy_dir,
        model_name=aligned_name,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        encoded_indices=encoded_indices,
    )

    print("\n" + "=" * 80)
    print("✅ SAE TRAINING + ENCODING COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - best_sae.pt: Best model checkpoint")
    print("  - training_history.json: Training metrics")
    print("  - sae_recon_error_hist.png: Recon error histogram")
    print(f"  - {base_name}_sae_encoded.npy: Encoded base activations {base_encoded.shape}")
    print(f"  - {aligned_name}_sae_encoded.npy: Encoded aligned activations {aligned_encoded.shape}")
    print(f"  - {args.model_family}_encoded_indices.npy: Row indices into pooled activations")
    print(f"\nNext step:")
    print(f"  python model/train_neuode_realtox.py --model_family {args.model_family}")


if __name__ == "__main__":
    main()

