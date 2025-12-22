#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np


def estimate_shape(fname, hidden_dim):
    """Estimate number of prompt vectors from file size."""
    size_bytes = os.path.getsize(fname)
    elems = size_bytes // np.dtype(np.float16).itemsize
    if elems % hidden_dim != 0:
        return None
    return elems // hidden_dim


def make_fixed_name(fname: Path) -> Path:
    """
    Convert:
        llama2_base_split_0_pooled.npy
    →   llama2_base_split_0_pooled_fixed.npy
    """
    stem = fname.stem  # removes .npy
    return fname.with_name(stem + "_fixed.npy")


def fix_single_file_safe(fname: Path, hidden_dim: int):
    print(f"\nProcessing: {fname}")

    n_prompts = estimate_shape(fname, hidden_dim)
    if n_prompts is None:
        print("  ⚠️  Warning: file size does not divide evenly by hidden_dim. Skipping.")
        return False

    print(f"  Detected shape: {n_prompts} × {hidden_dim}")

    # Load via memmap (raw fp16 buffer)
    raw = np.memmap(fname, dtype=np.float16, mode="r",
                    shape=(n_prompts, hidden_dim))

    # Convert to real numpy array (safe)
    arr = np.array(raw, dtype=np.float16)

    new_fname = make_fixed_name(fname)
    print(f"  → Writing SAFE fixed array to: {new_fname}")

    np.save(new_fname, arr)

    print("  ✓ Success — original file preserved")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing pooled .npy files")
    parser.add_argument("--hidden_dim", type=int, default=4096,
                        help="hidden dimension of pooled vectors (default 4096)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"\nScanning for pooled files under: {root}")
    print(f"Using hidden_dim = {args.hidden_dim}")

    pooled_files = list(root.rglob("*_pooled.npy"))
    print(f"Found {len(pooled_files)} pooled files")

    fixed_count = 0
    for f in pooled_files:
        if fix_single_file_safe(f, args.hidden_dim):
            fixed_count += 1

    print("\n==============================================")
    print("Finished SAFE conversion.")
    print(f"Successfully created {_plural(fixed_count,'file')} out of {len(pooled_files)} pooled files.")
    print("Original files remain untouched.")
    print("==============================================\n")


def _plural(n, word):
    return f"{n} {word}" + ("s" if n != 1 else "")


if __name__ == "__main__":
    main()
