"""
STEP 2: Train Neural-ODE Flow (Base → Aligned Transformation)

This script:
1. Loads SAE-encoded features from Step 1
2. Trains Neural-ODE to model base → aligned transformation
3. Computes action A(X) = ∫||v_θ||² dt for each sample
4. Identifies high-curvature regions (potential elicitation gaps)
5. Saves flow model and action values

Usage:
    python train_neural_ode.py --model_family llama2 --epochs 50
    
Requires:
    pip install torchdiffeq
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from torchdiffeq import odeint

# ============================================================================
# NEURAL ODE MODEL
# ============================================================================

class NeuralODEFunc(nn.Module):
    """
    Velocity field v_θ(x, t) for Neural ODE
    Models: dx/dt = v_θ(x, t)
    """
    
    def __init__(self, hidden_dim: int = 16384, time_dependent: bool = True):
        """
        Args:
            hidden_dim: Dimension of SAE features
            time_dependent: Whether v depends on time t
        """
        super().__init__()
        
        self.time_dependent = time_dependent
        
        # Velocity network
        if time_dependent:
            # Input: [x, t] where t is scalar time
            input_dim = hidden_dim + 1
        else:
            input_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, hidden_dim)
        )
        
        # Initialize with small weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x):
        """
        Args:
            t: Time (scalar or tensor)
            x: State [batch_size, hidden_dim]
        
        Returns:
            dx/dt: Velocity [batch_size, hidden_dim]
        """
        if self.time_dependent:
            # Expand time to match batch size
            if isinstance(t, (int, float)):
                t = torch.tensor([t], device=x.device, dtype=x.dtype)
            
            if len(t.shape) == 0:  # Scalar
                t = t.unsqueeze(0)
            
            t_expanded = t.expand(x.shape[0], 1)
            
            # Concatenate x and t
            xt = torch.cat([x, t_expanded], dim=1)
        else:
            xt = x
        
        return self.net(xt)

class NeuralODEFlow(nn.Module):
    """Complete Neural-ODE flow model"""
    
    def __init__(self, hidden_dim: int = 16384, time_dependent: bool = True):
        super().__init__()
        self.func = NeuralODEFunc(hidden_dim, time_dependent)
        self.hidden_dim = hidden_dim
    
    def forward(self, x0, t_span=None):
        """
        Integrate from t=0 to t=1
        
        Args:
            x0: Initial state [batch_size, hidden_dim]
            t_span: Time points to evaluate (default: [0, 1])
        
        Returns:
            x1: Final state at t=1
            trajectory: Full trajectory if t_span has multiple points
        """
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x0.device)
        
        # Integrate ODE
        trajectory = odeint(self.func, x0, t_span, method='dopri5')
        
        return trajectory
    
    def compute_action(self, x0, n_steps=10):
        """
        Compute action A(x) = ∫_0^1 ||v_θ(x(t), t)||² dt
        
        Args:
            x0: Initial state [batch_size, hidden_dim]
            n_steps: Number of integration steps
        
        Returns:
            action: Action values [batch_size]
        """
        t_span = torch.linspace(0, 1, n_steps, device=x0.device)
        
        # Get trajectory
        with torch.no_grad():
            trajectory = self.forward(x0, t_span)  # [n_steps, batch_size, hidden_dim]
        
        # Compute velocities at each point
        actions = []
        for i in range(len(t_span)):
            t = t_span[i]
            xt = trajectory[i]
            
            with torch.no_grad():
                v = self.func(t, xt)  # [batch_size, hidden_dim]
                v_norm_sq = (v ** 2).sum(dim=1)  # [batch_size]
                actions.append(v_norm_sq)
        
        # Integrate using trapezoidal rule
        actions = torch.stack(actions, dim=0)  # [n_steps, batch_size]
        dt = 1.0 / (n_steps - 1)
        action = torch.trapz(actions, dx=dt, dim=0)  # [batch_size]
        
        return action

# ============================================================================
# TRAINING
# ============================================================================

def train_neural_ode(model, train_loader, val_loader, device, epochs, lr, save_dir):
    """Train Neural-ODE flow"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_action': [],
        'val_action': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_action = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x_base, x_aligned in pbar:
            x_base = x_base.to(device)
            x_aligned = x_aligned.to(device)
            
            # Forward: integrate from base (t=0) to predicted aligned (t=1)
            t_span = torch.tensor([0.0, 1.0], device=device)
            trajectory = model(x_base, t_span)
            
            x_pred = trajectory[-1]  # Final state at t=1
            
            # Loss: MSE between predicted and actual aligned
            loss = nn.functional.mse_loss(x_pred, x_aligned)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute action (for monitoring)
            with torch.no_grad():
                action = model.compute_action(x_base, n_steps=5).mean()
            
            train_loss += loss.item()
            train_action += action.item()
            
            pbar.set_postfix({'loss': loss.item(), 'action': action.item()})
        
        train_loss /= len(train_loader)
        train_action /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_action = 0
        
        with torch.no_grad():
            for x_base, x_aligned in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x_base = x_base.to(device)
                x_aligned = x_aligned.to(device)
                
                t_span = torch.tensor([0.0, 1.0], device=device)
                trajectory = model(x_base, t_span)
                x_pred = trajectory[-1]
                
                loss = nn.functional.mse_loss(x_pred, x_aligned)
                action = model.compute_action(x_base, n_steps=5).mean()
                
                val_loss += loss.item()
                val_action += action.item()
        
        val_loss /= len(val_loader)
        val_action /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.6f}, Action: {train_action:.4f}")
        print(f"  Val   - Loss: {val_loss:.6f}, Action: {val_action:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_action'].append(train_action)
        history['val_action'].append(val_action)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'hidden_dim': model.hidden_dim
                }
            }
            torch.save(checkpoint, save_dir / 'best_neural_ode.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, save_dir / f'neural_ode_epoch_{epoch+1}.pt')
    
    # Save training history
    with open(save_dir / 'neural_ode_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

# ============================================================================
# COMPUTE ACTIONS FOR ALL SAMPLES
# ============================================================================

def compute_all_actions(model, base_encoded, device, batch_size=256, n_steps=10):
    """Compute action A(x) for all base samples"""
    
    print("\nComputing actions for all samples...")
    
    dataset = TensorDataset(torch.FloatTensor(base_encoded))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_actions = []
    
    with torch.no_grad():
        for (batch,) in tqdm(dataloader, desc="Computing actions"):
            batch = batch.to(device)
            actions = model.compute_action(batch, n_steps=n_steps)
            all_actions.append(actions.cpu().numpy())
    
    all_actions = np.concatenate(all_actions)
    
    print(f"Actions computed: {len(all_actions)}")
    print(f"  Mean: {all_actions.mean():.4f}")
    print(f"  Std:  {all_actions.std():.4f}")
    print(f"  Min:  {all_actions.min():.4f}")
    print(f"  Max:  {all_actions.max():.4f}")
    
    return all_actions

# ============================================================================
# ANALYSIS: IDENTIFY HIGH-CURVATURE REGIONS
# ============================================================================

def analyze_elicitation_gaps(actions, domains, variants, refusals, threshold_percentile=90):
    """Analyze where high action (curvature) occurs"""
    
    print("\n" + "="*80)
    print("ELICITATION GAP ANALYSIS")
    print("="*80)
    
    # Identify high-action samples (potential gaps)
    threshold = np.percentile(actions, threshold_percentile)
    high_action_mask = actions > threshold
    
    print(f"\nHigh-action threshold (>{threshold_percentile}th percentile): {threshold:.4f}")
    print(f"High-action samples: {high_action_mask.sum()} / {len(actions)} ({high_action_mask.mean():.1%})")
    
    # Analyze by domain
    print("\nBy Domain:")
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_high_action = high_action_mask & domain_mask
        print(f"  {domain}: {domain_high_action.sum()} / {domain_mask.sum()} "
              f"({domain_high_action.sum() / domain_mask.sum():.1%})")
    
    # Analyze by variant
    print("\nBy Variant:")
    unique_variants = np.unique(variants)
    for variant in sorted(unique_variants):
        variant_mask = variants == variant
        variant_high_action = high_action_mask & variant_mask
        print(f"  Variant {variant}: {variant_high_action.sum()} / {variant_mask.sum()} "
              f"({variant_high_action.sum() / variant_mask.sum():.1%})")
    
    # Analyze correlation with refusals
    if refusals is not None:
        print("\nRefusal Correlation:")
        refused_high_action = high_action_mask & refusals
        refused_low_action = (~high_action_mask) & refusals
        
        print(f"  High action + Refusal: {refused_high_action.sum()} "
              f"({refused_high_action.sum() / high_action_mask.sum():.1%} of high-action)")
        print(f"  Low action + Refusal: {refused_low_action.sum()} "
              f"({refused_low_action.sum() / (~high_action_mask).sum():.1%} of low-action)")
        
        # This suggests: Are refusals in high-curvature regions?
        if refused_high_action.sum() > refused_low_action.sum():
            print("\n  → Refusals ENRICHED in high-action regions (potential elicitation gaps)")
        else:
            print("\n  → Refusals NOT enriched in high-action regions")
    
    return {
        'threshold': threshold,
        'high_action_mask': high_action_mask,
        'high_action_count': high_action_mask.sum(),
        'by_domain': {domain: (high_action_mask & (domains == domain)).sum() 
                     for domain in unique_domains},
        'by_variant': {int(variant): (high_action_mask & (variants == variant)).sum() 
                      for variant in unique_variants}
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Neural-ODE Flow')
    parser.add_argument('--sae_dir', type=str, default='./sae_output', help='SAE output directory')
    parser.add_argument('--data_dir', type=str, default='./data', help='Original data directory')
    parser.add_argument('--model_family', type=str, default='llama2', choices=['llama2', 'falcon', 'mistral'])
    parser.add_argument('--output_dir', type=str, default='./ode_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--time_dependent', action='store_true', help='Use time-dependent velocity')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print(f"TRAINING NEURAL-ODE FLOW: {args.model_family}")
    print(f"{'='*80}\n")
    
    # Load SAE-encoded features
    sae_dir = Path(args.sae_dir)
    base_file = sae_dir / f"{args.model_family}_base_sae_encoded.npy"
    aligned_file = sae_dir / f"{args.model_family}_aligned_sae_encoded.npy"
    
    if not base_file.exists() or not aligned_file.exists():
        raise FileNotFoundError(
            f"SAE encoded files not found. Run train_sae.py first.\n"
            f"Looking for:\n  {base_file}\n  {aligned_file}"
        )
    
    print("Loading SAE-encoded features...")
    base_encoded = np.load(base_file)
    aligned_encoded = np.load(aligned_file)
    
    print(f"Base shape: {base_encoded.shape}")
    print(f"Aligned shape: {aligned_encoded.shape}")
    
    hidden_dim = base_encoded.shape[1]
    n_encoded = base_encoded.shape[0]
    
    # Load metadata (domains, variants, refusals)
    # We need to load this from original H5 files
    import h5py
    data_dir = Path(args.data_dir)
    h5_files = sorted(data_dir.glob(f"**/{args.model_family}_aligned_split_*.h5"))
    
    all_domains = []
    all_variants = []
    all_responses = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as hf:
            domains = [d.decode() if isinstance(d, bytes) else d for d in hf["domains"][:]]
            variants = hf["variants"][:].tolist()
            responses = [r.decode() if isinstance(r, bytes) else r for r in hf["responses"][:]]
            
            all_domains.extend(domains)
            all_variants.extend(variants)
            all_responses.extend(responses)
    
    all_domains = np.array(all_domains)
    all_variants = np.array(all_variants)
    
    # Detect refusals
    refusal_patterns = ["i cannot", "i can't", "i'm not able", "i'm sorry, but", "not appropriate"]
    refusals = np.array([any(p in r.lower() for p in refusal_patterns) for r in all_responses])
    
    print(f"Loaded {len(all_domains)} samples with metadata")
    print(f"Refusal rate: {refusals.mean():.1%}")
    
    # CRITICAL FIX: Subsample metadata to match encoded data
    # The encoded data was subsampled (10%), so we need to subsample metadata the same way
    if len(all_domains) != n_encoded:
        subsample_ratio = n_encoded / len(all_domains)
        print(f"\n⚠️  Metadata size ({len(all_domains)}) doesn't match encoded size ({n_encoded})")
        print(f"   Subsampling metadata by {subsample_ratio:.1%} to match encoded data...")
        
        # Use same subsampling pattern as encoding (take first N samples proportionally from each file)
        # Calculate how many samples per file
        samples_per_file = []
        for hf_file in h5_files:
            with h5py.File(hf_file, 'r') as hf:
                samples_per_file.append(len(hf["domains"]))
        
        # Subsample proportionally from each file (same as UltraFastActivationDataset)
        indices = []
        current_idx = 0
        for n_samples in samples_per_file:
            n_to_take = int(n_samples * subsample_ratio)
            indices.extend(range(current_idx, current_idx + n_to_take))
            current_idx += n_samples
        
        # Ensure we don't exceed the encoded size
        indices = indices[:n_encoded]
        
        # Apply subsampling
        all_domains = all_domains[indices]
        all_variants = all_variants[indices]
        refusals = refusals[indices]
        
        print(f"   ✓ Subsampled metadata to {len(all_domains)} samples (matches encoded data)")
    
    # Create dataset
    dataset = TensorDataset(
        torch.FloatTensor(base_encoded),
        torch.FloatTensor(aligned_encoded)
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Initialize Neural-ODE
    model = NeuralODEFlow(hidden_dim=hidden_dim, time_dependent=args.time_dependent).to(device)
    
    print(f"\nNeural-ODE Architecture:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Time-dependent: {args.time_dependent}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    
    history = train_neural_ode(model, train_loader, val_loader, device, args.epochs, args.lr, output_dir)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_neural_ode.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Compute actions for all samples
    print(f"\n{'='*80}")
    print("COMPUTING ACTIONS")
    print(f"{'='*80}")
    
    actions = compute_all_actions(model, base_encoded, device, batch_size=args.batch_size, n_steps=20)
    
    # Save actions
    np.save(output_dir / f"{args.model_family}_actions.npy", actions)
    
    # Analyze elicitation gaps
    gap_analysis = analyze_elicitation_gaps(actions, all_domains, all_variants, refusals)
    
    # Save analysis
    with open(output_dir / 'gap_analysis.json', 'w') as f:
        json.dump({
            'threshold': float(gap_analysis['threshold']),
            'high_action_count': int(gap_analysis['high_action_count']),
            'by_domain': {k: int(v) for k, v in gap_analysis['by_domain'].items()},
            'by_variant': {k: int(v) for k, v in gap_analysis['by_variant'].items()}
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ NEURAL-ODE TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - best_neural_ode.pt: Best model checkpoint")
    print(f"  - neural_ode_history.json: Training metrics")
    print(f"  - {args.model_family}_actions.npy: Action values for all samples")
    print(f"  - gap_analysis.json: Elicitation gap analysis")
    print(f"\nNext step:")
    print(f"  Run compute_optimal_transport.py to calculate Wasserstein distance")

if __name__ == "__main__":
    main()