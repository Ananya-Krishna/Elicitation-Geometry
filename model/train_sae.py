"""
STEP 1: Train Sparse Autoencoder on Base Model Activations

1. Loads base model activations from H5 files
2. Trains a sparse autoencoder to compress [4096] → [16384] (overcomplete)
3. Saves the trained SAE model
4. Encodes both base and aligned activations
5. Saves compressed representations for downstream analysis

Usage:
    python train_sae.py --model_family llama2 --epochs 20 --batch_size 256
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
import json
from typing import List, Tuple

# ============================================================================
# SPARSE AUTOENCODER MODEL
# ============================================================================

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for compressing LLM activations"""
    
    def __init__(self, 
                 input_dim: int = 4096,
                 hidden_dim: int = 16384,
                 sparsity_coef: float = 1e-3,
                 l1_coef: float = 1e-4):
        """
        Args:
            input_dim: Dimension of input activations (4096 for 7B models)
            hidden_dim: Dimension of sparse features (typically 4x input_dim)
            sparsity_coef: Weight for sparsity loss
            l1_coef: Weight for L1 regularization on features
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef
        self.l1_coef = l1_coef
        
        # Encoder: input → sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: sparse features → reconstructed input
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize with Xavier
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x):
        """Encode to sparse representation with ReLU activation"""
        return torch.relu(self.encoder(x))
    
    def decode(self, z):
        """Decode back to original space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(self, x, x_recon, z):
        """
        Compute total loss:
        1. Reconstruction loss (MSE)
        2. L1 sparsity loss on features
        3. Optional: KL divergence for sparsity constraint
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x)
        
        # L1 sparsity on features
        l1_loss = torch.mean(torch.abs(z))
        
        # Total loss
        total_loss = recon_loss + self.l1_coef * l1_loss
        
        # Compute sparsity metric (percentage of near-zero activations)
        sparsity = (torch.abs(z) < 0.01).float().mean()
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'l1': l1_loss,
            'sparsity': sparsity
        }

# ============================================================================
# DATASET FOR SAE TRAINING
# ============================================================================

class ActivationDataset(Dataset):
    """Load activations from your split H5 files for SAE training"""
    
    def __init__(self, 
                 data_dir: str,
                 model_name: str,  # e.g., "llama2_base"
                 layer_idx: int = -1,
                 pool_method: str = "mean",
                 max_samples: int = None):
        """
        Args:
            data_dir: Directory with split H5 files
            model_name: Model name (e.g., "llama2_base")
            layer_idx: Which layer to extract (-1 for last)
            pool_method: How to pool sequence dimension
            max_samples: Limit number of samples (for testing)
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.pool_method = pool_method
        
        # Find all split files
        split_files = sorted(self.data_dir.glob(f"{model_name}_split_*.h5"))
        
        if not split_files:
            raise ValueError(f"No split files found for {model_name} in {data_dir}")
        
        print(f"Found {len(split_files)} split files for {model_name}")
        
        # Load all splits
        self.h5_files = [h5py.File(f, 'r') for f in split_files]
        
        # Calculate sizes
        self.split_sizes = [h5["hidden_states"].shape[0] for h5 in self.h5_files]
        self.cumulative_sizes = np.cumsum([0] + self.split_sizes)
        self.total_size = self.cumulative_sizes[-1]
        
        # Limit samples if requested
        if max_samples is not None:
            self.total_size = min(self.total_size, max_samples)
        
        print(f"Total samples: {self.total_size:,}")
    
    def _get_split_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """Convert global index to (split_idx, local_idx)"""
        for split_idx in range(len(self.cumulative_sizes) - 1):
            if global_idx < self.cumulative_sizes[split_idx + 1]:
                local_idx = global_idx - self.cumulative_sizes[split_idx]
                return split_idx, local_idx
        raise IndexError(f"Index {global_idx} out of range")
    
    def _pool_sequence(self, hidden_states: np.ndarray) -> np.ndarray:
        """Pool sequence dimension [seq_len, hidden_dim] -> [hidden_dim]"""
        if self.pool_method == "mean":
            return hidden_states.mean(axis=0)
        elif self.pool_method == "max":
            return hidden_states.max(axis=0)
        elif self.pool_method == "last":
            # Get last non-zero token (simple heuristic)
            return hidden_states[-1]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """Return pooled activation vector"""
        split_idx, local_idx = self._get_split_and_local_idx(idx)
        
        # Get activation: [layers, seq_len, hidden_dim]
        hidden = self.h5_files[split_idx]["hidden_states"][local_idx]
        
        # Extract layer: [seq_len, hidden_dim]
        layer_hidden = hidden[self.layer_idx]
        
        # Pool: [hidden_dim]
        pooled = self._pool_sequence(layer_hidden)
        
        return torch.FloatTensor(pooled)
    
    def close(self):
        """Close all H5 files"""
        for h5 in self.h5_files:
            h5.close()

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_sae(model, train_loader, val_loader, device, epochs, lr, save_dir):
    """Train the Sparse Autoencoder"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_sparsity': [],
        'val_sparsity': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_metrics = {'total': 0, 'recon': 0, 'l1': 0, 'sparsity': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            
            # Forward
            x_recon, z = model(x)
            losses = model.compute_loss(x, x_recon, z)
            
            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            # Track metrics
            for key in train_metrics:
                train_metrics[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'sparsity': losses['sparsity'].item()
            })
        
        # Average training metrics
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation
        model.eval()
        val_metrics = {'total': 0, 'recon': 0, 'l1': 0, 'sparsity': 0}
        
        with torch.no_grad():
            for x in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x = x.to(device)
                x_recon, z = model(x)
                losses = model.compute_loss(x, x_recon, z)
                
                for key in val_metrics:
                    val_metrics[key] += losses[key].item()
        
        # Average validation metrics
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_metrics['total'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_metrics['total']:.6f}, "
              f"Recon: {train_metrics['recon']:.6f}, "
              f"L1: {train_metrics['l1']:.6f}, "
              f"Sparsity: {train_metrics['sparsity']:.2%}")
        print(f"  Val   - Loss: {val_metrics['total']:.6f}, "
              f"Recon: {val_metrics['recon']:.6f}, "
              f"L1: {val_metrics['l1']:.6f}, "
              f"Sparsity: {val_metrics['sparsity']:.2%}")
        
        # Save history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['train_sparsity'].append(train_metrics['sparsity'])
        history['val_sparsity'].append(val_metrics['sparsity'])
        
        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'config': {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'sparsity_coef': model.sparsity_coef,
                    'l1_coef': model.l1_coef
                }
            }
            torch.save(checkpoint, save_dir / 'best_sae.pt')
            print(f"  ✓ Saved best model (val_loss: {val_metrics['total']:.6f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, save_dir / f'sae_epoch_{epoch+1}.pt')
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

# ============================================================================
# ENCODING SCRIPT
# ============================================================================

def encode_and_save(sae, data_dir, model_name, output_dir, device, batch_size=256):
    """Encode all activations with trained SAE and save"""
    
    print(f"\nEncoding {model_name} activations...")
    
    # Load dataset
    dataset = ActivationDataset(data_dir, model_name, layer_idx=-1, pool_method="mean")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Encode
    sae.eval()
    encoded_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {model_name}"):
            batch = batch.to(device)
            z = sae.encode(batch)
            encoded_features.append(z.cpu().numpy())
    
    # Concatenate
    encoded_features = np.vstack(encoded_features)
    
    print(f"Encoded shape: {encoded_features.shape}")
    print(f"Sparsity: {(np.abs(encoded_features) < 0.01).mean():.2%}")
    
    # Save
    output_file = output_dir / f"{model_name}_sae_encoded.npy"
    np.save(output_file, encoded_features)
    print(f"Saved to: {output_file}")
    
    dataset.close()
    return encoded_features

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Sparse Autoencoder')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--model_family', type=str, default='llama2', choices=['llama2', 'falcon', 'mistral'])
    parser.add_argument('--output_dir', type=str, default='./sae_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=16384, help='SAE hidden dimension')
    parser.add_argument('--l1_coef', type=float, default=1e-4, help='L1 regularization coefficient')
    parser.add_argument('--train_on_base', action='store_true', help='Train on base model only')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine which model to train on
    if args.train_on_base:
        train_model = f"{args.model_family}_base"
        print(f"\n{'='*80}")
        print(f"Training SAE on BASE model only: {train_model}")
        print(f"{'='*80}\n")
    else:
        train_model = f"{args.model_family}_base"
        print(f"\n{'='*80}")
        print(f"Training SAE on: {train_model}")
        print(f"Will encode both base and aligned after training")
        print(f"{'='*80}\n")
    
    # Load training data
    train_dataset = ActivationDataset(args.data_dir, train_model, layer_idx=-1)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_subset):,}")
    print(f"Val samples: {len(val_subset):,}")
    
    # Get input dimension from first sample
    sample = train_dataset[0]
    input_dim = sample.shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Initialize SAE
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        l1_coef=args.l1_coef
    ).to(device)
    
    print(f"\nSAE Architecture:")
    print(f"  Input: {input_dim}")
    print(f"  Hidden: {args.hidden_dim}")
    print(f"  Compression ratio: {args.hidden_dim / input_dim:.2f}x (overcomplete)")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    
    # Train
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    
    history = train_sae(sae, train_loader, val_loader, device, args.epochs, args.lr, output_dir)
    
    train_dataset.close()
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_sae.pt')
    sae.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Encode both base and aligned
    print(f"\n{'='*80}")
    print("ENCODING ACTIVATIONS")
    print(f"{'='*80}\n")
    
    base_encoded = encode_and_save(sae, args.data_dir, f"{args.model_family}_base", 
                                   output_dir, device, args.batch_size)
    aligned_encoded = encode_and_save(sae, args.data_dir, f"{args.model_family}_aligned", 
                                      output_dir, device, args.batch_size)
    
    print(f"\n{'='*80}")
    print("✅ SAE TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - best_sae.pt: Best model checkpoint")
    print(f"  - training_history.json: Training metrics")
    print(f"  - {args.model_family}_base_sae_encoded.npy: Encoded base activations")
    print(f"  - {args.model_family}_aligned_sae_encoded.npy: Encoded aligned activations")
    print(f"\nNext steps:")
    print(f"  1. Run train_neural_ode.py to fit flow between base and aligned")
    print(f"  2. Run compute_optimal_transport.py to calculate W2 distance")

if __name__ == "__main__":
    main()