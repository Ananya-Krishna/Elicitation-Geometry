"""
Encode the remaining 50% of data points and combine with existing encodings.

This script:
1. Loads existing encodings (first 50% = 2,250 samples)
2. Encodes the remaining 50% (samples 2,250-4,499)
3. Combines them into full encodings (4,500 samples total)
4. Saves to the same location (overwrites with full dataset)

IMPORTANT: This assumes the existing encodings are from the FIRST 50% of data.
The subsampling in UltraFastActivationDataset takes samples sequentially from the start.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.train_sawe_realtox import UltraFastActivationDataset, SparseAutoencoder


class SecondHalfDataset(UltraFastActivationDataset):
    """
    Modified dataset that takes the SECOND half of samples (indices 2250-4499).
    This is done by modifying the indices to skip the first 50% and take the next 50%.
    """
    
    def __init__(self, *args, **kwargs):
        # Temporarily set subsample to 1.0 to get all indices
        kwargs['subsample'] = 1.0
        
        # Call parent to get all data loaded
        super().__init__(*args, **kwargs)
        
        # Now take only the second half
        total_samples = len(self.data)
        half_point = total_samples // 2
        
        # Verify we have the expected total
        if total_samples != 4500:
            print(f"⚠️  Warning: Expected 4,500 total samples, got {total_samples}")
        
        # Take samples from half_point to end (second half)
        self.data = self.data[half_point:]
        
        print(f"✓ Using SECOND half: samples {half_point:,} to {total_samples:,} ({len(self.data):,} samples)")


def encode_remaining_half(
    sae_path: Path,
    data_dir: str,
    model_name: str,
    output_dir: Path,
    device,
    batch_size: int = 256,
):
    """
    Encode the remaining 50% of data and combine with existing encodings.
    """
    print(f"\n{'='*80}")
    print(f"ENCODING REMAINING HALF: {model_name}")
    print(f"{'='*80}")
    
    # Check if existing encoding exists
    existing_file = output_dir / f"{model_name}_sae_encoded.npy"
    if not existing_file.exists():
        raise FileNotFoundError(
            f"Existing encoding not found: {existing_file}\n"
            "Please ensure the first half has been encoded."
        )
    
    # Load existing encoding (first 50%)
    existing_encoded = np.load(existing_file)
    print(f"✓ Loaded existing encoding: {existing_encoded.shape}")
    print(f"  (First half: {existing_encoded.shape[0]:,} samples)")
    
    if existing_encoded.shape[0] != 2250:
        print(f"⚠️  Warning: Expected 2,250 samples in existing encoding, got {existing_encoded.shape[0]}")
    
    # Load trained SAE
    print(f"\nLoading SAE from: {sae_path}")
    checkpoint = torch.load(sae_path, map_location=device)
    
    # Determine input_dim from checkpoint
    if 'model_state_dict' in checkpoint:
        encoder_weight = checkpoint['model_state_dict']['encoder.weight']
        input_dim = encoder_weight.shape[1]
        hidden_dim = encoder_weight.shape[0]
    elif 'config' in checkpoint:
        # Use config if available
        config = checkpoint['config']
        input_dim = config.get('input_dim', 4096)
        hidden_dim = config.get('hidden_dim', 16384)
    else:
        # Fallback: try to infer from the model
        input_dim = 4096  # Default
        hidden_dim = 16384  # Default
    
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # Handle both old and new checkpoint formats
    if 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'])
    else:
        sae.load_state_dict(checkpoint)
    
    sae.to(device)
    sae.eval()
    print(f"✓ Loaded SAE: {input_dim} → {hidden_dim}")
    
    # Create dataset for SECOND half
    print(f"\nLoading SECOND half of data...")
    dataset = SecondHalfDataset(
        data_dir=data_dir,
        model_name=model_name,
        layer_idx=-1,
        pool_method="mean",
        subsample=1.0,  # Will be overridden to take second half
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    
    # Encode second half
    print(f"\nEncoding second half ({len(dataset):,} samples)...")
    sae.eval()
    encoded_chunks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {model_name} (second half)"):
            batch = batch.to(device, non_blocking=True)
            z = sae.encode(batch)
            encoded_chunks.append(z.cpu().numpy())
    
    second_half_encoded = np.vstack(encoded_chunks)
    sparsity = (np.abs(second_half_encoded) < 0.01).mean()
    
    print(f"  Encoded shape: {second_half_encoded.shape}")
    print(f"  Sparsity: {sparsity:.2%}")
    
    # Combine with existing (first half)
    print(f"\nCombining encodings...")
    full_encoded = np.vstack([existing_encoded, second_half_encoded])
    print(f"  First half: {existing_encoded.shape}")
    print(f"  Second half: {second_half_encoded.shape}")
    print(f"  Combined: {full_encoded.shape}")
    
    # Verify we have the full dataset
    expected_total = 4500
    if full_encoded.shape[0] != expected_total:
        print(f"⚠️  Warning: Expected {expected_total} samples total, got {full_encoded.shape[0]}")
    
    # Save combined encoding (overwrites existing)
    output_file = output_dir / f"{model_name}_sae_encoded.npy"
    np.save(output_file, full_encoded)
    print(f"\n✓ Saved FULL encoding to: {output_file}")
    print(f"  Shape: {full_encoded.shape}")
    print(f"  Total samples: {full_encoded.shape[0]:,}")
    
    dataset.close()
    return full_encoded


def main():
    parser = argparse.ArgumentParser(
        description="Encode remaining 50% of data and combine with existing encodings"
    )
    parser.add_argument(
        "--sae_dir",
        type=str,
        required=True,
        help="Directory containing trained SAE (best_sae.pt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing H5 split files",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        choices=["llama2", "falcon", "mistral"],
        help="Model family",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing existing encodings (will be updated)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    sae_dir = Path(args.sae_dir)
    data_dir = args.data_dir
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    
    # Try model-specific SAE first, then fall back to generic best_sae.pt
    model_specific_sae = sae_dir / f"{args.model_family}_best_sae.pt"
    generic_sae = sae_dir / "best_sae.pt"
    
    if model_specific_sae.exists():
        sae_path = model_specific_sae
        print(f"✓ Using model-specific SAE: {sae_path}")
    elif generic_sae.exists():
        sae_path = generic_sae
        print(f"⚠️  Using generic SAE (may not match model dimensions): {sae_path}")
        print(f"   If you get dimension errors, ensure {model_specific_sae} exists")
    else:
        raise FileNotFoundError(
            f"SAE checkpoint not found. Tried:\n"
            f"  - {model_specific_sae}\n"
            f"  - {generic_sae}"
        )
    
    print(f"Device: {device}")
    print(f"SAE: {sae_path}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    
    # Encode both base and aligned
    base_name = f"{args.model_family}_base"
    aligned_name = f"{args.model_family}_aligned"
    
    print(f"\n{'='*80}")
    print(f"ENCODING REMAINING HALF FOR {args.model_family.upper()}")
    print(f"{'='*80}")
    
    # Base
    print(f"\n[1/2] Encoding {base_name}...")
    encode_remaining_half(
        sae_path=sae_path,
        data_dir=data_dir,
        model_name=base_name,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
    )
    
    # Aligned
    print(f"\n[2/2] Encoding {aligned_name}...")
    encode_remaining_half(
        sae_path=sae_path,
        data_dir=data_dir,
        model_name=aligned_name,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
    )
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE: Full encodings saved for {args.model_family}")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - {base_name}_sae_encoded.npy: {4500} samples")
    print(f"  - {aligned_name}_sae_encoded.npy: {4500} samples")


if __name__ == "__main__":
    main()

