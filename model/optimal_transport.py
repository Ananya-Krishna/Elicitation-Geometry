"""
STEP 3: Compute Optimal Transport & Complete Geometric Analysis

This script:
1. Computes Wasserstein-2 distance between base and aligned distributions
2. Compares OT distance with Neural-ODE action
3. Identifies discrepancies (curvature signatures)
4. Generates visualizations and final report
5. Tests cross-domain transfer

Usage:
    python compute_optimal_transport.py --model_family llama2
    
Requires:
    pip install POT matplotlib seaborn scikit-learn
"""

import numpy as np
import torch
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import ot  # Python Optimal Transport library

# ============================================================================
# OPTIMAL TRANSPORT COMPUTATION
# ============================================================================

def compute_wasserstein_distance(base_features, aligned_features, n_samples=5000, reg=0.01):
    """
    Compute Wasserstein-2 distance using entropic regularization
    
    Args:
        base_features: [N, D] base model features
        aligned_features: [N, D] aligned model features
        n_samples: Number of samples to use (subsample for efficiency)
        reg: Entropic regularization parameter
    
    Returns:
        w2_distance: Wasserstein-2 distance
        ot_plan: Optimal transport plan (coupling)
    """
    print("\n" + "="*80)
    print("COMPUTING OPTIMAL TRANSPORT")
    print("="*80)
    
    # Subsample for computational efficiency
    if len(base_features) > n_samples:
        indices = np.random.choice(len(base_features), n_samples, replace=False)
        base_sub = base_features[indices]
        aligned_sub = aligned_features[indices]
    else:
        base_sub = base_features
        aligned_sub = aligned_features
    
    print(f"\nUsing {len(base_sub):,} samples")
    print(f"Feature dimension: {base_sub.shape[1]}")
    
    # Uniform distributions
    a = np.ones(len(base_sub)) / len(base_sub)
    b = np.ones(len(aligned_sub)) / len(aligned_sub)
    
    # Compute cost matrix (squared Euclidean distance)
    print("\nComputing cost matrix...")
    M = ot.dist(base_sub, aligned_sub, metric='sqeuclidean')
    print(f"Cost matrix shape: {M.shape}")
    print(f"Cost matrix stats: min={M.min():.2f}, max={M.max():.2f}, mean={M.mean():.2f}")
    
    # Compute Wasserstein distance with entropic regularization
    print(f"\nComputing Wasserstein-2 distance (reg={reg})...")
    w2_sinkhorn = ot.sinkhorn2(a, b, M, reg)
    
    print(f"✓ Wasserstein-2 distance: {w2_sinkhorn:.6f}")
    
    # Also compute OT plan for analysis
    print("\nComputing optimal transport plan...")
    ot_plan = ot.sinkhorn(a, b, M, reg)
    
    return float(w2_sinkhorn), ot_plan, indices if len(base_features) > n_samples else None

# ============================================================================
# COMPARE OT WITH NEURAL-ODE ACTION
# ============================================================================

def compare_ot_and_action(w2_distance, actions, output_dir):
    """
    Compare Wasserstein distance with mean Neural-ODE action
    Discrepancy suggests curvature in the transformation
    """
    print("\n" + "="*80)
    print("COMPARING OT DISTANCE vs NEURAL-ODE ACTION")
    print("="*80)
    
    mean_action = actions.mean()
    std_action = actions.std()
    
    print(f"\nWasserstein-2 distance: {w2_distance:.6f}")
    print(f"Mean ODE Action:        {mean_action:.6f}")
    print(f"Std ODE Action:         {std_action:.6f}")
    
    # Interpretation
    ratio = mean_action / w2_distance if w2_distance > 0 else 0
    print(f"\nAction / W2 ratio: {ratio:.4f}")
    
    if ratio > 1.2:
        print("  → HIGH curvature detected!")
        print("     The flow takes a 'curved path', suggesting complex warping")
        print("     Potential elicitation gaps in regions of high action")
    elif ratio < 0.8:
        print("  → LOW curvature detected")
        print("     The flow is nearly straight (minimal warping)")
    else:
        print("  → MODERATE curvature")
        print("     Some warping present but not extreme")
    
    results = {
        'w2_distance': w2_distance,
        'mean_action': mean_action,
        'std_action': std_action,
        'action_w2_ratio': ratio
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(base_features, aligned_features, actions, domains, 
                         refusals, output_dir, n_viz=2000):
    """Create visualization plots"""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Subsample for visualization
    if len(base_features) > n_viz:
        indices = np.random.choice(len(base_features), n_viz, replace=False)
        base_viz = base_features[indices]
        aligned_viz = aligned_features[indices]
        actions_viz = actions[indices]
        domains_viz = domains[indices] if len(domains) > 0 else None
        refusals_viz = refusals[indices] if len(refusals) > 0 else None
    else:
        base_viz = base_features
        aligned_viz = aligned_features
        actions_viz = actions
        domains_viz = domains
        refusals_viz = refusals
    
    # 1. Action distribution
    print("\n1. Plotting action distribution...")
    plt.figure(figsize=(10, 6))
    plt.hist(actions, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(actions.mean(), color='r', linestyle='--', label=f'Mean: {actions.mean():.4f}')
    plt.axvline(np.percentile(actions, 90), color='orange', linestyle='--', 
                label=f'90th percentile: {np.percentile(actions, 90):.4f}')
    plt.xlabel('Action A(x)')
    plt.ylabel('Count')
    plt.title('Distribution of Neural-ODE Action Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / 'action_distribution.png', dpi=150)
    plt.close()
    
    # 2. Actions by domain
    if domains_viz is not None and len(domains_viz) > 0:
        print("2. Plotting actions by domain...")
        plt.figure(figsize=(10, 6))
        unique_domains = np.unique(domains)
        domain_actions = [actions[domains == d] for d in unique_domains]
        plt.boxplot(domain_actions, labels=unique_domains)
        plt.ylabel('Action A(x)')
        plt.title('Action Distribution by Domain')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'actions_by_domain.png', dpi=150)
        plt.close()
    
    # 3. Actions vs Refusals
    if refusals_viz is not None and len(refusals_viz) > 0:
        print("3. Plotting actions vs refusals...")
        plt.figure(figsize=(10, 6))
        refused_actions = actions[refusals]
        not_refused_actions = actions[~refusals]
        plt.hist(not_refused_actions, bins=50, alpha=0.5, label='No Refusal', color='blue')
        plt.hist(refused_actions, bins=50, alpha=0.5, label='Refusal', color='red')
        plt.xlabel('Action A(x)')
        plt.ylabel('Count')
        plt.title('Action Distribution: Refusal vs No Refusal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / 'actions_vs_refusals.png', dpi=150)
        plt.close()
    
    # 4. t-SNE visualization
    print("4. Computing t-SNE embedding...")
    # Combine base and aligned for joint embedding
    combined = np.vstack([base_viz, aligned_viz])
    labels = np.array(['Base'] * len(base_viz) + ['Aligned'] * len(aligned_viz))
    
    # Use PCA first for speedup
    if combined.shape[1] > 50:
        pca = PCA(n_components=50)
        combined_pca = pca.fit_transform(combined)
    else:
        combined_pca = combined
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined_pca)
    
    plt.figure(figsize=(12, 6))
    
    # Plot base vs aligned
    plt.subplot(1, 2, 1)
    for label in ['Base', 'Aligned']:
        mask = labels == label
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   label=label, alpha=0.5, s=10)
    plt.title('t-SNE: Base vs Aligned')
    plt.legend()
    
    # Plot colored by action (base only)
    plt.subplot(1, 2, 2)
    base_embedded = embedded[:len(base_viz)]
    scatter = plt.scatter(base_embedded[:, 0], base_embedded[:, 1], 
                         c=actions_viz, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Action')
    plt.title('t-SNE: Colored by Action')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'tsne_visualization.png', dpi=150)
    plt.close()
    
    print(f"\n✓ Visualizations saved to {viz_dir}")

# ============================================================================
# CROSS-DOMAIN ANALYSIS
# ============================================================================

def analyze_cross_domain_transfer(actions, domains, refusals):
    """Analyze if patterns transfer across domains"""
    
    print("\n" + "="*80)
    print("CROSS-DOMAIN TRANSFER ANALYSIS")
    print("="*80)
    
    unique_domains = np.unique(domains)
    
    # Compare action distributions across domains
    print("\nAction statistics by domain:")
    domain_stats = {}
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_actions = actions[domain_mask]
        domain_refusals = refusals[domain_mask] if len(refusals) > 0 else None
        
        stats = {
            'count': int(domain_mask.sum()),
            'mean_action': float(domain_actions.mean()),
            'std_action': float(domain_actions.std()),
            'high_action_pct': float((domain_actions > np.percentile(actions, 90)).mean()),
        }
        
        if domain_refusals is not None:
            stats['refusal_rate'] = float(domain_refusals.mean())
            # Correlation between action and refusal in this domain
            if len(np.unique(domain_refusals)) > 1:  # Check if there's variation
                corr, pval = spearmanr(domain_actions, domain_refusals)
                stats['action_refusal_corr'] = float(corr)
                stats['action_refusal_pval'] = float(pval)
        
        domain_stats[domain] = stats
        
        print(f"\n  {domain}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    
    return domain_stats

# ============================================================================
# GENERATE FINAL REPORT
# ============================================================================

def generate_report(w2_distance, actions, domains, refusals, ot_comparison, 
                   domain_stats, output_dir):
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("="*80)
    report.append("ELICITATION GAP GEOMETRIC ANALYSIS - FINAL REPORT")
    report.append("="*80)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-" * 80)
    report.append(f"Total samples analyzed: {len(actions):,}")
    report.append(f"Wasserstein-2 distance: {w2_distance:.6f}")
    report.append(f"Mean Neural-ODE Action: {actions.mean():.6f}")
    report.append(f"Action / W2 ratio: {ot_comparison['action_w2_ratio']:.4f}")
    report.append("")
    
    # High-action regions
    high_action_threshold = np.percentile(actions, 90)
    high_action_count = (actions > high_action_threshold).sum()
    report.append(f"High-action samples (>90th percentile): {high_action_count} ({high_action_count/len(actions):.1%})")
    report.append("")
    
    # Domain analysis
    report.append("DOMAIN ANALYSIS")
    report.append("-" * 80)
    for domain, stats in domain_stats.items():
        report.append(f"\n{domain}:")
        report.append(f"  Samples: {stats['count']}")
        report.append(f"  Mean action: {stats['mean_action']:.6f}")
        report.append(f"  High-action rate: {stats['high_action_pct']:.1%}")
        if 'refusal_rate' in stats:
            report.append(f"  Refusal rate: {stats['refusal_rate']:.1%}")
        if 'action_refusal_corr' in stats:
            report.append(f"  Action-Refusal correlation: {stats['action_refusal_corr']:.3f} (p={stats['action_refusal_pval']:.4f})")
    report.append("")
    
    # Interpretation
    report.append("INTERPRETATION")
    report.append("-" * 80)
    
    if ot_comparison['action_w2_ratio'] > 1.2:
        report.append("✓ ELICITATION GAPS DETECTED")
        report.append("  The Neural-ODE flow exhibits high curvature, suggesting that")
        report.append("  alignment has warped the representation space in complex ways.")
        report.append("  Regions of high action correspond to potential elicitation gaps")
        report.append("  where model competence is suppressed but latently present.")
    else:
        report.append("✓ MINIMAL WARPING DETECTED")
        report.append("  The transformation from base to aligned is relatively direct.")
        report.append("  Alignment appears to preserve most of the geometry.")
    
    # Check refusal correlation
    if len(refusals) > 0 and len(np.unique(refusals)) > 1:
        overall_corr, overall_pval = spearmanr(actions, refusals)
        report.append("")
        report.append(f"Overall Action-Refusal correlation: {overall_corr:.3f} (p={overall_pval:.4f})")
        if overall_corr > 0.1 and overall_pval < 0.05:
            report.append("  → Refusals are significantly correlated with high action")
            report.append("  → This supports the elicitation gap hypothesis")
    
    report.append("")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(output_dir / 'final_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Report saved to {output_dir / 'final_report.txt'}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute Optimal Transport and Complete Analysis')
    parser.add_argument('--sae_dir', type=str, default='./sae_output', help='SAE output directory')
    parser.add_argument('--ode_dir', type=str, default='./ode_output', help='Neural-ODE output directory')
    parser.add_argument('--data_dir', type=str, default='./data', help='Original data directory')
    parser.add_argument('--model_family', type=str, default='llama2', choices=['llama2', 'falcon', 'mistral'])
    parser.add_argument('--output_dir', type=str, default='./analysis_output', help='Output directory')
    parser.add_argument('--ot_samples', type=int, default=5000, help='Samples for OT computation')
    parser.add_argument('--ot_reg', type=float, default=0.01, help='OT entropic regularization')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print(f"GEOMETRIC ANALYSIS: {args.model_family}")
    print(f"{'='*80}\n")
    
    # Load SAE-encoded features
    sae_dir = Path(args.sae_dir)
    base_file = sae_dir / f"{args.model_family}_base_sae_encoded.npy"
    aligned_file = sae_dir / f"{args.model_family}_aligned_sae_encoded.npy"
    
    print("Loading SAE-encoded features...")
    base_features = np.load(base_file)
    aligned_features = np.load(aligned_file)
    print(f"  Base: {base_features.shape}")
    print(f"  Aligned: {aligned_features.shape}")
    
    # Load Neural-ODE actions
    ode_dir = Path(args.ode_dir)
    actions_file = ode_dir / f"{args.model_family}_actions.npy"
    
    print("\nLoading Neural-ODE actions...")
    actions = np.load(actions_file)
    print(f"  Actions: {actions.shape}")
    
    # Load metadata
    import h5py
    data_dir = Path(args.data_dir)
    h5_files = sorted(data_dir.glob(f"{args.model_family}_aligned_split_*.h5"))
    
    print("\nLoading metadata from H5 files...")
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
    refusal_patterns = ["i cannot", "i can't", "i'm not able", "i'm sorry, but", "not appropriate", "against my guidelines"]
    refusals = np.array([any(p in r.lower() for p in refusal_patterns) for r in all_responses])
    
    print(f"  Domains: {len(all_domains)}")
    print(f"  Unique domains: {np.unique(all_domains)}")
    print(f"  Refusal rate: {refusals.mean():.1%}")
    
    # Step 1: Compute Optimal Transport
    w2_distance, ot_plan, ot_indices = compute_wasserstein_distance(
        base_features, aligned_features, 
        n_samples=args.ot_samples, 
        reg=args.ot_reg
    )
    
    # Save OT results
    ot_results = {
        'w2_distance': float(w2_distance),
        'n_samples': args.ot_samples,
        'regularization': args.ot_reg
    }
    with open(output_dir / 'ot_results.json', 'w') as f:
        json.dump(ot_results, f, indent=2)
    
    # Step 2: Compare OT with Neural-ODE action
    ot_comparison = compare_ot_and_action(w2_distance, actions, output_dir)
    
    with open(output_dir / 'ot_comparison.json', 'w') as f:
        json.dump(ot_comparison, f, indent=2)
    
    # Step 3: Cross-domain analysis
    domain_stats = analyze_cross_domain_transfer(actions, all_domains, refusals)
    
    with open(output_dir / 'domain_analysis.json', 'w') as f:
        json.dump(domain_stats, f, indent=2)
    
    # Step 4: Create visualizations
    create_visualizations(base_features, aligned_features, actions, 
                         all_domains, refusals, output_dir)
    
    # Step 5: Generate final report
    generate_report(w2_distance, actions, all_domains, refusals, 
                   ot_comparison, domain_stats, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE GEOMETRIC ANALYSIS FINISHED")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - final_report.txt: Comprehensive analysis report")
    print(f"  - ot_results.json: Optimal transport results")
    print(f"  - ot_comparison.json: OT vs Neural-ODE comparison")
    print(f"  - domain_analysis.json: Cross-domain statistics")
    print(f"  - visualizations/: All plots")
    print(f"\n🎉 Your S&DS 689 project analysis is complete!")

if __name__ == "__main__":
    main()