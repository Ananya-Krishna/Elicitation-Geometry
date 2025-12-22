#!/usr/bin/env python3
"""
Convert HDF5 LLM activations → NumPy features for RealToxicityPrompts.

Fixes:
  - Avoids recursive globbing that accidentally picks up old_data_backup.
  - Streams HDF5 rows one-by-one instead of loading entire hidden_states (prevents OOM / "Killed").
  - Uses explicit elicitation_data_{model_key} directories under a data_root.

Assumes:
  data_root/
    elicitation_data_llama2_base/
      llama2_base_split_0_activations.h5
      llama2_base_split_1_activations.h5
      ...
    elicitation_data_llama2_aligned/
      llama2_aligned_split_0_activations.h5
      llama2_aligned_split_1_activations.h5
      ...

Usage example:
  python convert_hdf5_to_numpy_realtox.py \
      --data_root /nfs/.../Elicitation-Geometry/data \
      --model_family llama2 \
      --output_dir ./numpy_data \
      --layer_idx -1 \
      --pool_method mean
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm


def discover_split_files(data_root: Path, model_family: str, which: str):
    """
    Discover HDF5 split files for base or aligned models in a NON-recursive way.

    which: 'base' or 'aligned'
      - 'base'    → elicitation_data_llama2_base / llama2_base_split_*.h5
      - 'aligned' → elicitation_data_llama2_aligned / llama2_aligned_split_*.h5
    """
    assert which in {"base", "aligned"}
    model_key = f"{model_family}_{which}"
    elic_dir = data_root / f"elicitation_data_{model_key}"

    if not elic_dir.exists():
        raise FileNotFoundError(f"Elicitation dir not found: {elic_dir}")

    # Non-recursive glob: ONLY look inside this specific directory.
    split_files = sorted(elic_dir.glob(f"{model_key}_split_*_activations.h5"))

    if not split_files:
        raise FileNotFoundError(
            f"No HDF5 files found in {elic_dir} for model_key={model_key}"
        )

    print(f"Found {len(split_files)} HDF5 files for {model_key}:")
    for f in split_files:
        print(f"  - {f}")

    return split_files


def pool_sequence(hidden_states: np.ndarray, pool_method: str) -> np.ndarray:
    """
    hidden_states: [seq_len, hidden_dim]
    returns: [hidden_dim]
    """
    if pool_method == "mean":
        return hidden_states.mean(axis=0)
    elif pool_method == "max":
        return hidden_states.max(axis=0)
    elif pool_method == "last":
        return hidden_states[-1]
    else:
        raise ValueError(f"Unknown pool_method: {pool_method}")


def stream_pool_h5_files(
    split_files,
    layer_idx: int = -1,
    pool_method: str = "mean",
    max_samples: int | None = None,
):
    """
    Stream over a list of HDF5 activation files, row-by-row, and pool each row
    over the sequence dimension at the specified layer.

    Returns:
      features: np.ndarray [N, hidden_dim]
    """
    pooled_list = []
    total_rows = 0

    # First pass: figure out total rows for progress reporting (optional)
    for fpath in split_files:
        with h5py.File(fpath, "r") as hf:
            total_rows += hf["hidden_states"].shape[0]

    if max_samples is not None:
        total_rows = min(total_rows, max_samples)

    print(f"Total rows across splits (capped): {total_rows}")

    processed = 0
    pbar = tqdm(total=total_rows, desc="Pooling activations")

    for fpath in split_files:
        with h5py.File(fpath, "r") as hf:
            hs_ds = hf["hidden_states"]  # shape: [rows, layers, seq_len, dim]
            n_rows, n_layers, seq_len, dim = hs_ds.shape

            # Interpret negative layer_idx like Python indexing
            layer = layer_idx if layer_idx >= 0 else (n_layers + layer_idx)

            if not (0 <= layer < n_layers):
                raise ValueError(
                    f"Invalid layer_idx {layer_idx} for dataset with {n_layers} layers"
                )

            for i in range(n_rows):
                if max_samples is not None and processed >= max_samples:
                    break

                # Read ONE row: shape [layers, seq_len, dim]
                row = hs_ds[i]  # this stays small enough for memory
                layer_hidden = row[layer]  # [seq_len, dim]
                pooled = pool_sequence(layer_hidden, pool_method)
                pooled_list.append(pooled.astype(np.float32))

                processed += 1
                pbar.update(1)

            if max_samples is not None and processed >= max_samples:
                break

    pbar.close()

    features = np.stack(pooled_list, axis=0)
    print(f"Pooled features shape: {features.shape}  (~{features.nbytes/1e6:.1f} MB)")

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 activations → NumPy (RealToxicityPrompts, streaming & safe)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root dir containing elicitation_data_{model_key} folders.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="llama2",
        choices=["llama2", "falcon", "mistral"],
        help="Model family (prefix before _base / _aligned).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./numpy_data",
        help="Where to save .npy feature arrays.",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="Layer index to pool (Python-style, -1 = last layer).",
    )
    parser.add_argument(
        "--pool_method",
        type=str,
        default="mean",
        choices=["mean", "max", "last"],
        help="How to pool over the sequence dimension.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on total samples for quick tests. Use None for all.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CONVERTING HDF5 → NUMPY (RealToxicityPrompts) for {args.model_family}")
    print("=" * 80)

    # -------------------- BASE MODEL --------------------
    print(f"\nPooling activations for {args.model_family}_base...")
    base_files = discover_split_files(data_root, args.model_family, which="base")
    base_features = stream_pool_h5_files(
        base_files,
        layer_idx=args.layer_idx,
        pool_method=args.pool_method,
        max_samples=args.max_samples,
    )
    base_out = output_dir / f"{args.model_family}_base_features.npy"
    np.save(base_out, base_features)
    print(f"Saved base features to: {base_out}")

    # -------------------- ALIGNED MODEL --------------------
    print(f"\nPooling activations for {args.model_family}_aligned...")
    aligned_files = discover_split_files(data_root, args.model_family, which="aligned")
    aligned_features = stream_pool_h5_files(
        aligned_files,
        layer_idx=args.layer_idx,
        pool_method=args.pool_method,
        max_samples=args.max_samples,
    )
    aligned_out = output_dir / f"{args.model_family}_aligned_features.npy"
    np.save(aligned_out, aligned_features)
    print(f"Saved aligned features to: {aligned_out}")

    print("\n✅ HDF5 → NumPy conversion complete.")
    print(f"Base   features: {base_features.shape}")
    print(f"Aligned features: {aligned_features.shape}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()

