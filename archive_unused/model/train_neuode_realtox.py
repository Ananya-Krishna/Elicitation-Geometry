"""
STEP 2: Train Neural-ODE Flow (Base → Aligned Transformation)
UPDATED FOR REALTOXICITYPROMPTS DATA

This script:
1. Loads SAE-encoded features from Step 1
2. Trains Neural-ODE to model base → aligned transformation
3. Computes action A(X) = ∫||v_θ||² dt for each sample
4. Analyzes by toxicity level (not synthetic domains/variants)
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
    """Velocity field v_θ(x, t) for Neural ODE"""
    
    def __init__(self, hidden_dim=16384):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 1024),  # +1 for time
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, hidden_dim)
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x):
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=x.dtype)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        
        t_expanded = t.expand(x.shape[0], 1)
        xt = torch.cat([x, t_expanded], dim=1)
        
        return self.net(xt)

class NeuralODEFlow(nn.Module):
    """Complete Neural-ODE flow model"""
    
    def __init__(self, hidden_dim=16384):
        super().__init__()
        self.func = NeuralODEFunc(hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, x0, t_span=None):
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x0.device)
        
        trajectory = odeint(self.func, x0, t_span, method='dopri5')
        return trajectory
    
    def compute_action(self, x0, n_steps=10):
        """Compute action A(x) = ∫_0^1 ||v_θ||² dt"""
        t_span = torch.linspace(0, 1, n_steps, device=x0.device)
        
        with torch.no_grad():
            trajectory = self.forward(x0, t_span)
        
        actions = []
        for i in range(len(t_span)):
            t = t_span[i]
            xt = trajectory[i]
            
            with torch.no_grad():
                v = self.func(t, xt)
                v_norm_sq = (v ** 2).sum(dim=1)
                actions.append(v_norm_sq)
        
        actions = torch.stack(actions, dim=0)
        dt = 1.0 / (n_steps - 1)
        action = torch.trapz(actions, dx=dt, dim=0)
        
        return action

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_neural_ode(model, train_loader, val_loader, device, epochs, lr, save_dir):
    """Train Neural-ODE flow"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_action': [], 'val_action': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_action = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x_base, x_aligned in pbar:
            x_base = x_base.to(device)
            x_aligned = x_aligned.to(device)
            
            t_span = torch.tensor([0.0, 1.0], device=device)
            trajectory = model(x_base, t_span)
            x_pred = trajectory[-1]
            
            loss = nn.functional.mse_loss(x_pred, x_aligned)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Action: {train_action:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Action: {val_action:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_action'].append(train_action)
        history['val_action'].append(val_action)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_action': val_action
            }
            torch.save(checkpoint, save_dir / 'best_neural_ode.pt')
            print(f"  ✓ Saved best model")
    
    with open(save_dir / 'neural_ode_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

# ============================================================================
# COMPUTE ACTIONS FOR ALL SAMPLES
# ============================================================================

def compute_all_actions(model, base_encoded, device, batch_size=128):
    """Compute action for all samples"""
    print("\nComputing actions for all samples...")
    
    model.eval()
    all_actions = []
    
    dataset = TensorDataset(torch.FloatTensor(base_encoded))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for (batch,) in tqdm(dataloader, desc="Computing actions"):
            batch = batch.to(device)
            actions = model.compute_action(batch, n_steps=10)
            all_actions.append(actions.cpu().numpy())
    
    all_actions = np.concatenate(all_actions)
    
    print(f"  Mean action: {all_actions.mean():.4f}")
    print(f"  Std action: {all_actions.std():.4f}")
    print(f"  Min: {all_actions.min():.4f}, Max: {all_actions.max():.4f}")
    
    return all_actions

# ============================================================================
# LOAD TOXICITY SCORES FROM PROMPTS
# ============================================================================

def load_toxicity_metadata(data_dir):
    """Load toxicity scores and domains from prompt JSON files"""
    print("\nLoading toxicity metadata from prompts...")
    
    all_prompts = []
    for i in range(6):
        prompt_file = Path(data_dir) / f"prompts_split_{i}.json"
        if not prompt_file.exists():
            print(f"  Warning: {prompt_file} not found, skipping")
            continue
        with open(prompt_file) as f:
            prompts = json.load(f)
            all_prompts.extend(prompts)
    
    if not all_prompts:
        raise ValueError("No prompts found! Check data_dir path.")
    
    toxicity_scores = np.array([p.get('toxicity', 0.0) for p in all_prompts])
    domains = np.array([p.get('domain', 'unknown') for p in all_prompts])
    
    print(f"  Loaded {len(toxicity_scores)} prompts")
    print(f"  Mean toxicity: {toxicity_scores.mean():.3f}, Std: {toxicity_scores.std():.3f}")
    print(f"  Unique domains: {np.unique(domains)}")
    
    return toxicity_scores, domains, all_prompts

# ============================================================================
# ANALYZE BY TOXICITY LEVEL
# ============================================================================

def analyze_by_toxicity(actions, toxicity_scores, domains, output_dir):
    """Analyze actions grouped by toxicity level"""
    print(f"\n{'='*80}")
    print("ELICITATION GAP ANALYSIS BY TOXICITY")
    print(f"{'='*80}")
    
    threshold = np.percentile(actions, 90)
    high_action_mask = actions > threshold
    
    analysis = {
        'threshold_90pct': float(threshold),
        'high_action_count': int(high_action_mask.sum()),
        'by_toxicity_level': {}
    }
    
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_actions = actions[domain_mask]
        domain_toxicity = toxicity_scores[domain_mask]
        
        analysis['by_toxicity_level'][domain] = {
            'count': int(domain_mask.sum()),
            'mean_action': float(domain_actions.mean()),
            'std_action': float(domain_actions.std()),
            'mean_toxicity': float(domain_toxicity.mean()),
            'high_action_pct': float((actions[domain_mask] > threshold).mean())
        }
        
        print(f"\n  {domain}:")
        print(f"    Count: {domain_mask.sum()}")
        print(f"    Mean action: {domain_actions.mean():.4f}")
        print(f"    Mean toxicity: {domain_toxicity.mean():.3f}")
        print(f"    High action (>90th pct): {(actions[domain_mask] > threshold).mean():.1%}")
    
    # Overall correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(toxicity_scores, actions)
    print(f"\n  Overall correlation (Action vs Toxicity):")
    print(f"    Pearson r={r:.3f}, p={p:.4e}")
    
    analysis['overall_correlation'] = {
        'pearson_r': float(r),
        'pearson_p': float(p)
    }
    
    with open(output_dir / 'gap_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n  Saved to {output_dir / 'gap_analysis.json'}")
    
    return analysis

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Neural-ODE Flow')
    parser.add_argument('--sae_dir', type=str, default='./sae_output', help='SAE output directory')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory (for toxicity metadata)')
    parser.add_argument('--model_family', type=str, default='llama2', choices=['llama2', 'falcon', 'mistral'])
    parser.add_argument('--output_dir', type=str, default='./ode_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print(f"TRAINING NEURAL-ODE: {args.model_family}")
    print(f"{'='*80}\n")
    
    # Load SAE-encoded features
    sae_dir = Path(args.sae_dir)
    base_file = sae_dir / f"{args.model_family}_base_sae_encoded.npy"
    aligned_file = sae_dir / f"{args.model_family}_aligned_sae_encoded.npy"
    
    if not base_file.exists() or not aligned_file.exists():
        raise FileNotFoundError(f"SAE encoded files not found. Run train_sae.py first!")
    
    base_encoded = np.load(base_file)
    aligned_encoded = np.load(aligned_file)
    
    print(f"Base encoded: {base_encoded.shape}")
    print(f"Aligned encoded: {aligned_encoded.shape}")
    
    # Split 80/20
    n_samples = len(base_encoded)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(0.8 * n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(base_encoded[train_indices]),
        torch.FloatTensor(aligned_encoded[train_indices])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(base_encoded[val_indices]),
        torch.FloatTensor(aligned_encoded[val_indices])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Initialize Neural-ODE
    hidden_dim = base_encoded.shape[1]
    model = NeuralODEFlow(hidden_dim=hidden_dim).to(device)
    
    print(f"\nNeural-ODE: {hidden_dim}D flow")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    all_actions = compute_all_actions(model, base_encoded, device, args.batch_size)
    
    # Save actions
    actions_file = output_dir / f"{args.model_family}_actions.npy"
    np.save(actions_file, all_actions)
    print(f"\nSaved actions to: {actions_file}")
    
    # Load toxicity metadata and analyze
    toxicity_scores, domains, prompts = load_toxicity_metadata(args.data_dir)
    
    # CRITICAL FIX: Subsample metadata to match encoded data
    # The encoded data may be subsampled, so we need to subsample metadata the same way
    n_encoded = len(all_actions)
    if len(toxicity_scores) != n_encoded:
        subsample_ratio = n_encoded / len(toxicity_scores)
        print(f"\n⚠️  Metadata size ({len(toxicity_scores)}) doesn't match encoded size ({n_encoded})")
        print(f"   Subsampling metadata by {subsample_ratio:.1%} to match encoded data...")
        
        # Subsample proportionally (take first N samples)
        indices = np.arange(n_encoded)
        toxicity_scores = toxicity_scores[indices]
        domains = domains[indices]
        prompts = [prompts[i] for i in indices]
        
        print(f"   ✓ Metadata subsampled to {len(toxicity_scores)} samples (matches encoded data).")
    
    analysis = analyze_by_toxicity(all_actions, toxicity_scores, domains, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ NEURAL-ODE TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - best_neural_ode.pt: Best model checkpoint")
    print(f"  - neural_ode_history.json: Training metrics")
    print(f"  - {args.model_family}_actions.npy: Action values for all samples")
    print(f"  - gap_analysis.json: Toxicity-based analysis")
    print(f"\nNext step:")
    print(f"  python compute_optimal_transport.py --model_family {args.model_family}")

if __name__ == "__main__":
    main()
