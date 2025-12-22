"""
STEP 1: Train Sparse Autoencoder on Base Model Activations
ULTRA-FAST VERSION (Optimized for ~2-hour completion)

This script:
1. Loads base model activations from split H5 files (RealToxicityPrompts format)
2. Trains a sparse autoencoder [4096] → [16384] (overcomplete, sparse)
3. Saves trained SAE model + training history
4. Encodes both base and aligned activations (optionally subsampled)
5. Saves compressed representations

Key optimizations:
- Keeps H5 files open (no open/close per sample)
- Preloads pooled activations into RAM (fast random access)
- Subsampling for training and encoding to control runtime

Usage:
    python train_sae_ultra.py \
        --model_family llama2 \
        --epochs 5 \
        --batch_size 512 \
        --subsample 0.5 \
        --encode_subsample 0.1
"""

import argparse
import json
from pathlib import Path

import h5py
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
        input_dim: int = 4096,
        hidden_dim: int = 16384,
        sparsity_coef: float = 1e-3,  # kept for compatibility / future use
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
# ULTRA-OPTIMIZED DATASET (keeps files open + preloads pooled activations)
# ============================================================================

class UltraFastActivationDataset(Dataset):
    """
    ULTRA-OPTIMIZED: Keeps H5 files open and preloads pooled activations into memory.

    - Reads from split H5 files with "hidden_states" dataset.
    - Pools sequence dimension at a given layer.
    - Supports subsampling and max_samples for runtime control.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        layer_idx: int = -1,
        pool_method: str = "mean",
        max_samples: int | None = None,
        subsample: float = 1.0,
    ):
        """
        Args:
            data_dir: Directory containing split H5 files.
            model_name: e.g., "llama2_base".
            layer_idx: Which layer to extract (-1 for last).
            pool_method: "mean", "max", or "last" pooling over sequence.
            max_samples: Optional cap on number of samples.
            subsample: Fraction of data to use (0.5 = 50%).
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.pool_method = pool_method

        # ---- File discovery (compatible with both old/new naming patterns) ----
        # Primary pattern (from original script 1)
        split_files = sorted(
            self.data_dir.glob(f"{model_name}_split_*_activations.h5")
        )

        # Fallback: recursive search with same pattern
        if not split_files:
            split_files = sorted(
                self.data_dir.glob(f"**/{model_name}_split_*_activations.h5")
            )

        # Fallback: more general pattern (from script 2)
        if not split_files:
            split_files = sorted(self.data_dir.glob(f"**/{model_name}_split_*.h5"))

        if not split_files:
            raise ValueError(f"No split files found for {model_name} in {data_dir}")

        print(f"Found {len(split_files)} split files for {model_name}")
        self.split_files = split_files

        # ---- Keep files open ----
        print("Opening H5 files (keeping them open for fast access)...")
        self.h5_files = []
        self.split_sizes = []
        for f in split_files:
            h5 = h5py.File(f, "r")
            self.h5_files.append(h5)
            self.split_sizes.append(h5["hidden_states"].shape[0])

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

        # ---- Preload all pooled activations into memory ----
        print(
            f"Pre-loading ALL {len(self.indices):,} pooled samples into memory "
            "(fast access)..."
        )
        data_list = []
        for split_idx, local_idx in tqdm(self.indices, desc="Loading data"):
            hidden = self.h5_files[split_idx]["hidden_states"][local_idx]
            layer_hidden = hidden[self.layer_idx]  # [seq_len, hidden_dim]
            pooled = self._pool_sequence(layer_hidden)
            data_list.append(pooled.astype(np.float32))

        self.data = np.stack(data_list, axis=0)
        print(
            f"✓ Pre-loaded {self.data.shape[0]:,} samples "
            f"({self.data.nbytes / 1e6:.1f} MB)"
        )

    def _pool_sequence(self, hidden_states: np.ndarray) -> np.ndarray:
        """Pool [seq_len, hidden_dim] → [hidden_dim]."""
        if self.pool_method == "mean":
            return hidden_states.mean(axis=0)
        elif self.pool_method == "max":
            return hidden_states.max(axis=0)
        elif self.pool_method == "last":
            return hidden_states[-1]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])

    def close(self):
        """Close all H5 files and free memory."""
        for h5 in self.h5_files:
            h5.close()
        self.h5_files = []
        self.data = None


# ============================================================================
# TRAINING LOOP (functionality of script 1 + efficiency tweaks)
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

        # Periodic checkpoints every 5 epochs (if we have a checkpoint already)
        if checkpoint is not None and (epoch + 1) % 5 == 0:
            torch.save(checkpoint, save_dir / f"sae_epoch_{epoch + 1}.pt")

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ============================================================================
# ENCODING SCRIPT (with subsampling + sparsity stats)
# ============================================================================

def encode_and_save(
    sae,
    data_dir: str,
    model_name: str,
    output_dir: Path,
    device,
    batch_size: int = 256,
    subsample: float = 1.0,
):
    """
    Encode activations using trained SAE and save to .npy

    subsample < 1.0 lets you encode only a fraction of data to save time.
    """
    print(f"\nEncoding {model_name} activations...")
    print(f"  Using subsample = {subsample * 100:.1f}%")

    dataset = UltraFastActivationDataset(
        data_dir=data_dir,
        model_name=model_name,
        layer_idx=-1,
        pool_method="mean",
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {model_name}"):
            batch = batch.to(device, non_blocking=True)
            z = sae.encode(batch)
            encoded_chunks.append(z.cpu().numpy())

    encoded_features = np.vstack(encoded_chunks)
    sparsity = (np.abs(encoded_features) < 0.01).mean()

    print(f"  Encoded shape: {encoded_features.shape}")
    print(f"  Sparsity: {sparsity:.2%}")

    output_file = output_dir / f"{model_name}_sae_encoded.npy"
    np.save(output_file, encoded_features)
    print(f"  Saved to: {output_file}")

    dataset.close()
    return encoded_features


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ULTRA-FAST SAE Training (2-hour target)"
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
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
        help="Use only this fraction of data for training (0.5 = 50%)",
    )
    parser.add_argument(
        "--encode_subsample",
        type=float,
        default=0.1,
        help=(
            "Use only this fraction of data for encoding "
            "(0.1 = 10%%, CRITICAL for speed)"
        ),
    )
    parser.add_argument(
        "--train_on_base",
        action="store_true",
        help="(Reserved) Train on base model only (default behavior)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_model = f"{args.model_family}_base"

    print("\n" + "=" * 80)
    print(f"ULTRA-FAST SAE Training: {train_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Subsample (train): {args.subsample * 100:.1f}%")
    print("=" * 80 + "\n")

    # --------------------- LOAD DATASET (ULTRA-FAST) ---------------------
    dataset = UltraFastActivationDataset(
        data_dir=args.data_dir,
        model_name=train_model,
        layer_idx=-1,
        pool_method="mean",
        max_samples=None,
        subsample=args.subsample,
    )

    # Split train/val (80/20)
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

    print(f"Train samples: {len(train_subset):,}")
    print(f"Val samples: {len(val_subset):,}")

    # Get input dimension from a sample
    sample = dataset[0]
    input_dim = sample.shape[0]
    print(f"Input dimension: {input_dim}")

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
    print("TRAINING")
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

    # Free dataset H5 handles (preloaded data still in memory, but we don't need it)
    dataset.close()

    # --------------------- LOAD BEST MODEL ---------------------
    checkpoint = torch.load(output_dir / "best_sae.pt", map_location=device)
    sae.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")

    # --------------------- ENCODING ---------------------
    print("\n" + "=" * 80)
    print("ENCODING ACTIVATIONS (SUBSAMPLED FOR SPEED)")
    print("=" * 80 + "\n")
    print(
        f"⚠️  Encoding only {args.encode_subsample * 100:.1f}% of data for each model to save time!"
    )
    print("   (Set --encode_subsample 1.0 if you need full encoding.)\n")

    base_name = f"{args.model_family}_base"
    aligned_name = f"{args.model_family}_aligned"

    base_encoded = encode_and_save(
        sae=sae,
        data_dir=args.data_dir,
        model_name=base_name,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        subsample=args.encode_subsample,
    )

    aligned_encoded = encode_and_save(
        sae=sae,
        data_dir=args.data_dir,
        model_name=aligned_name,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        subsample=args.encode_subsample,
    )

    print("\n" + "=" * 80)
    print("✅ SAE TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - best_sae.pt: Best model checkpoint")
    print("  - training_history.json: Training metrics")
    print(f"  - {base_name}_sae_encoded.npy: Encoded base activations "
          f"({base_encoded.shape})")
    print(f"  - {aligned_name}_sae_encoded.npy: Encoded aligned activations "
          f"({aligned_encoded.shape})")
    print(f"\nNext step:")
    print(f"  python train_neural_ode.py --model_family {args.model_family}")


if __name__ == "__main__":
    main()

