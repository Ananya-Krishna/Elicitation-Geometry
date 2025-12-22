"""
ULTRA-FAST SAE Training - Optimized for 2-hour completion
- Keeps H5 files open (no open/close per sample)
- Uses regular gpu_h200 partition (better I/O)
- Reduced epochs and batch optimization
- Chunked data loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# ============================================================================
# SPARSE AUTOENCODER MODEL (same as before)
# ============================================================================

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for compressing LLM activations"""
    
    def __init__(self, 
                 input_dim: int = 4096,
                 hidden_dim: int = 16384,
                 sparsity_coef: float = 1e-3,
                 l1_coef: float = 1e-4):
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
    
    def encode(self, x):
        return torch.relu(self.encoder(x))
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(self, x, x_recon, z):
        recon_loss = nn.functional.mse_loss(x_recon, x)
        l1_loss = torch.mean(torch.abs(z))
        total_loss = recon_loss + self.l1_coef * l1_loss
        sparsity = (torch.abs(z) < 0.01).float().mean()
        return {
            'total': total_loss,
            'recon': recon_loss,
            'l1': l1_loss,
            'sparsity': sparsity
        }

# ============================================================================
# ULTRA-OPTIMIZED DATASET - KEEPS FILES OPEN
# ============================================================================

class UltraFastActivationDataset(Dataset):
    """
    ULTRA-OPTIMIZED: Keeps H5 files open, reads in chunks
    This eliminates the open/close overhead that kills performance on NFS
    """
    
    def __init__(self, 
                 data_dir: str,
                 model_name: str,
                 layer_idx: int = -1,
                 pool_method: str = "mean",
                 max_samples: int = None,
                 subsample: float = 1.0):
        """
        Args:
            subsample: Use only this fraction of data (0.5 = 50% of samples)
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.pool_method = pool_method
        
        # Find all split files
        split_files = sorted(self.data_dir.glob(f"**/{model_name}_split_*.h5"))
        if not split_files:
            raise ValueError(f"No split files found for {model_name} in {data_dir}")
        
        print(f"Found {len(split_files)} split files for {model_name}")
        self.split_files = split_files
        
        # CRITICAL: Keep files open (don't close them)
        print("Opening H5 files (keeping them open for fast access)...")
        self.h5_files = []
        self.split_sizes = []
        
        for f in split_files:
            h5 = h5py.File(f, 'r')  # Keep open!
            self.h5_files.append(h5)
            self.split_sizes.append(h5["hidden_states"].shape[0])
        
        self.cumulative_sizes = np.cumsum([0] + self.split_sizes)
        self.total_size = self.cumulative_sizes[-1]
        
        # Apply subsampling
        if subsample < 1.0:
            self.total_size = int(self.total_size * subsample)
            print(f"Subsampling to {self.total_size:,} samples ({subsample*100:.0f}%)")
        
        if max_samples is not None:
            self.total_size = min(self.total_size, max_samples)
        
        print(f"Total samples: {self.total_size:,}")
        
        # Pre-compute indices for fast access
        self.indices = []
        for split_idx in range(len(self.split_sizes)):
            start_idx = self.cumulative_sizes[split_idx]
            end_idx = min(self.cumulative_sizes[split_idx + 1], 
                         self.cumulative_sizes[0] + self.total_size)
            for local_idx in range(end_idx - start_idx):
                self.indices.append((split_idx, local_idx))
        
        # CRITICAL FIX: Pre-load ALL data into memory
        # With 50% subsampling, this is only ~36MB - totally manageable
        print(f"Pre-loading ALL {len(self.indices):,} samples into memory (fast access)...")
        self.data = []
        for i in tqdm(range(len(self.indices)), desc="Loading data"):
            split_idx, local_idx = self.indices[i]
            hidden = self.h5_files[split_idx]["hidden_states"][local_idx]
            layer_hidden = hidden[self.layer_idx]
            pooled = self._pool_sequence(layer_hidden)
            self.data.append(pooled.astype(np.float32))
        
        # Convert to numpy array for fast indexing
        self.data = np.array(self.data)
        print(f"✓ Pre-loaded {len(self.data):,} samples ({self.data.nbytes / 1e6:.1f} MB)")
    
    def _pool_sequence(self, hidden_states: np.ndarray) -> np.ndarray:
        if self.pool_method == "mean":
            return hidden_states.mean(axis=0)
        elif self.pool_method == "max":
            return hidden_states.max(axis=0)
        elif self.pool_method == "last":
            return hidden_states[-1]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Fast access - data is pre-loaded in memory!"""
        # Use pre-loaded data (no H5 I/O!)
        return torch.FloatTensor(self.data[idx])
    
    def close(self):
        """Close all H5 files and free memory"""
        for h5 in self.h5_files:
            h5.close()
        self.h5_files = []
        del self.data
        self.data = None

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_sae(model, train_loader, val_loader, device, epochs, lr, save_dir):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x in pbar:
            x = x.to(device, non_blocking=True)
            
            x_recon, z = model(x)
            losses = model.compute_loss(x, x_recon, z)
            
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            train_loss += losses['total'].item()
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'sparsity': losses['sparsity'].item()
            })
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for x in val_pbar:
                x = x.to(device, non_blocking=True)
                x_recon, z = model(x)
                losses = model.compute_loss(x, x_recon, z)
                val_loss += losses['total'].item()
                val_pbar.set_postfix({'loss': losses['total'].item()})
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_sae.pt')
            print(f"✓ Saved best model (val_loss={val_loss:.4f})")
    
    return model

def encode_and_save(sae, data_dir, model_name, output_dir, device, batch_size, subsample=0.1):
    """
    Encode activations using trained SAE
    
    CRITICAL OPTIMIZATION: Only encode a small fraction (default 10%) to save time.
    This is the main bottleneck - encoding all 4,500 samples takes ~1.5 hours per file!
    """
    print(f"Encoding {model_name} activations (subsampling to {subsample*100:.0f}% for speed)...")
    dataset = UltraFastActivationDataset(
        data_dir, 
        model_name, 
        layer_idx=-1, 
        pool_method="mean",
        subsample=subsample  # CRITICAL: Only encode 10% of data!
    )
    
    sae.eval()
    encoded_features = []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            batch = batch.to(device, non_blocking=True)
            z = sae.encode(batch)
            encoded_features.append(z.cpu().numpy())
    
    encoded_features = np.vstack(encoded_features)
    output_file = output_dir / f"{model_name}_sae_encoded.npy"
    np.save(output_file, encoded_features)
    print(f"Saved to: {output_file} (shape: {encoded_features.shape})")
    
    dataset.close()
    return encoded_features

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ULTRA-FAST SAE Training (2-hour target)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--model_family', type=str, default='llama2', choices=['llama2', 'falcon', 'mistral'])
    parser.add_argument('--output_dir', type=str, default='./sae_output_ultra', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (reduced for speed)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size (increased for speed)')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate (increased for faster convergence)')
    parser.add_argument('--hidden_dim', type=int, default=16384, help='SAE hidden dimension')
    parser.add_argument('--l1_coef', type=float, default=1e-4, help='L1 regularization coefficient')
    parser.add_argument('--subsample', type=float, default=0.5, help='Use only this fraction of data for training (0.5 = 50%%)')
    parser.add_argument('--encode_subsample', type=float, default=0.1, help='Use only this fraction of data for encoding (0.1 = 10%%, CRITICAL for speed)')
    parser.add_argument('--train_on_base', action='store_true', help='Train on base model only')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    train_model = f"{args.model_family}_base"
    print(f"\n{'='*80}")
    print(f"ULTRA-FAST SAE Training: {train_model}")
    print(f"  Epochs: {args.epochs} (reduced)")
    print(f"  Batch size: {args.batch_size} (increased)")
    print(f"  Subsample: {args.subsample*100:.0f}% of data")
    print(f"{'='*80}\n")
    
    # Load dataset (ULTRA-FAST version - keeps files open)
    dataset = UltraFastActivationDataset(
        args.data_dir, 
        train_model, 
        layer_idx=-1, 
        max_samples=None,
        subsample=args.subsample
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True, persistent_workers=False)
    
    print(f"Train samples: {len(train_subset):,}")
    print(f"Val samples: {len(val_subset):,}")
    
    # Get input dimension
    sample = dataset[0]
    input_dim = sample.shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        l1_coef=args.l1_coef
    ).to(device)
    
    print(f"\nSAE Architecture:")
    print(f"  Input: {input_dim}")
    print(f"  Hidden: {args.hidden_dim}")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    
    # Train
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    
    train_sae(sae, train_loader, val_loader, device, args.epochs, args.lr, output_dir)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_sae.pt')
    sae.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Encode both base and aligned (HEAVILY SUBSAMPLED for speed)
    print(f"\n{'='*80}")
    print("ENCODING ACTIVATIONS (SUBSAMPLED FOR SPEED)")
    print(f"{'='*80}\n")
    print(f"⚠️  Encoding only {args.encode_subsample*100:.0f}% of data to save time!")
    print(f"   (Full encoding would take ~9 hours - this takes ~{args.encode_subsample*9:.1f} hours)\n")
    
    base_encoded = encode_and_save(sae, args.data_dir, f"{args.model_family}_base", 
                                   output_dir, device, args.batch_size, 
                                   subsample=args.encode_subsample)
    aligned_encoded = encode_and_save(sae, args.data_dir, f"{args.model_family}_aligned", 
                                      output_dir, device, args.batch_size,
                                      subsample=args.encode_subsample)
    
    dataset.close()
    print(f"\n✓ Complete! Outputs saved to: {output_dir}")

if __name__ == '__main__':
    main()

