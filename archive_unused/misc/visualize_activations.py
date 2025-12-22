#!/usr/bin/env python3
"""
Hidden Activation Visualization Script
Visualizes hidden activations from LLaMA models (base vs aligned) for a single prompt

This script provides both regular and sparse visualizations of hidden activations,
comparing base and aligned models to understand the differences in internal representations.

Usage:
    python visualize_activations.py --model_family llama2 --prompt_idx 0 --layer_idx -1
    python visualize_activations.py --model_family llama2 --prompt_idx 5 --layer_idx 10 --sparse_threshold 0.1
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ActivationVisualizer:
    """Visualize hidden activations from LLaMA models"""
    
    def __init__(self, data_dir: str = "./elicitation_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("./activation_visualizations")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_model_data(self, model_name: str, split_id: int) -> Dict[str, Any]:
        """Load activation data for a specific model"""
        data_dir = Path(f"elicitation_data_{model_name}")
        h5_file = data_dir / f"{model_name}_split_{split_id}_activations.h5"
        prompts_file = data_dir / f"prompts_split_{split_id}.json"
        
        if not h5_file.exists():
            raise FileNotFoundError(f"Activation file not found: {h5_file}")
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
        
        # Load prompts
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        
        # Load HDF5 data
        with h5py.File(h5_file, 'r') as f:
            data = {
                'hidden_states': f['hidden_states'][:],  # [n_samples, n_layers, seq_len, hidden_dim]
                'responses': [r.decode() if isinstance(r, bytes) else r for r in f['responses'][:]],
                'domains': [d.decode() if isinstance(d, bytes) else d for d in f['domains'][:]],
                'variants': f['variants'][:],
                'prompt_ids': [p.decode() if isinstance(p, bytes) else p for p in f['prompt_ids'][:]],
                'prompts': prompts
            }
        
        return data
    
    def get_activation_for_prompt(self, data: Dict[str, Any], prompt_idx: int, layer_idx: int) -> Tuple[np.ndarray, str, str]:
        """Extract activation for a specific prompt and layer"""
        hidden_states = data['hidden_states'][prompt_idx]  # [n_layers, seq_len, hidden_dim]
        layer_activation = hidden_states[layer_idx]  # [seq_len, hidden_dim]
        prompt_text = data['prompts'][prompt_idx]['prompt']
        response_text = data['responses'][prompt_idx]
        
        return layer_activation, prompt_text, response_text
    
    def create_regular_visualizations(self, base_activation: np.ndarray, aligned_activation: np.ndarray,
                                    prompt_text: str, response_base: str, response_aligned: str,
                                    layer_idx: int, prompt_idx: int) -> None:
        """Create regular (non-sparse) visualizations"""
        
        print(f"\n📊 Creating regular visualizations for prompt {prompt_idx}, layer {layer_idx}")
        
        # 1. Activation heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Hidden Activations: Base vs Aligned (Layer {layer_idx})', fontsize=16)
        
        # Base model heatmap
        im1 = axes[0, 0].imshow(base_activation.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[0, 0].set_title('Base Model Activations')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Hidden Dimension')
        plt.colorbar(im1, ax=axes[0, 0], label='Activation Value')
        
        # Aligned model heatmap
        im2 = axes[0, 1].imshow(aligned_activation.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[0, 1].set_title('Aligned Model Activations')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Hidden Dimension')
        plt.colorbar(im2, ax=axes[0, 1], label='Activation Value')
        
        # Difference heatmap
        diff_activation = aligned_activation - base_activation
        im3 = axes[1, 0].imshow(diff_activation.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[1, 0].set_title('Difference (Aligned - Base)')
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Hidden Dimension')
        plt.colorbar(im3, ax=axes[1, 0], label='Activation Difference')
        
        # Activation magnitude comparison
        base_magnitude = np.linalg.norm(base_activation, axis=1)
        aligned_magnitude = np.linalg.norm(aligned_activation, axis=1)
        axes[1, 1].plot(base_magnitude, label='Base', alpha=0.7)
        axes[1, 1].plot(aligned_magnitude, label='Aligned', alpha=0.7)
        axes[1, 1].set_title('Activation Magnitude by Position')
        axes[1, 1].set_xlabel('Sequence Position')
        axes[1, 1].set_ylabel('L2 Norm')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'regular_activations_prompt_{prompt_idx}_layer_{layer_idx}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Activation Distributions: Base vs Aligned (Layer {layer_idx})', fontsize=16)
        
        # Flatten activations for distribution analysis
        base_flat = base_activation.flatten()
        aligned_flat = aligned_activation.flatten()
        
        # Histogram comparison
        axes[0, 0].hist(base_flat, bins=50, alpha=0.6, label='Base', density=True, color='blue')
        axes[0, 0].hist(aligned_flat, bins=50, alpha=0.6, label='Aligned', density=True, color='red')
        axes[0, 0].set_title('Activation Value Distribution')
        axes[0, 0].set_xlabel('Activation Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[0, 1].boxplot([base_flat, aligned_flat], labels=['Base', 'Aligned'])
        axes[0, 1].set_title('Activation Value Distribution (Box Plot)')
        axes[0, 1].set_ylabel('Activation Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot: base vs aligned
        # Subsample for visualization if too many points
        if len(base_flat) > 10000:
            indices = np.random.choice(len(base_flat), 10000, replace=False)
            base_sample = base_flat[indices]
            aligned_sample = aligned_flat[indices]
        else:
            base_sample = base_flat
            aligned_sample = aligned_flat
        
        axes[1, 0].scatter(base_sample, aligned_sample, alpha=0.3, s=1)
        axes[1, 0].plot([base_sample.min(), base_sample.max()], [base_sample.min(), base_sample.max()], 
                       'r--', alpha=0.8, label='y=x')
        axes[1, 0].set_title('Base vs Aligned Activations')
        axes[1, 0].set_xlabel('Base Activation Value')
        axes[1, 0].set_ylabel('Aligned Activation Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation plot
        correlation = np.corrcoef(base_flat, aligned_flat)[0, 1]
        axes[1, 1].text(0.5, 0.5, f'Correlation: {correlation:.4f}', 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Activation Correlation')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'distributions_prompt_{prompt_idx}_layer_{layer_idx}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Prompt and response display
        self._create_prompt_response_plot(prompt_text, response_base, response_aligned, 
                                        prompt_idx, layer_idx)
    
    def create_sparse_visualizations(self, base_activation: np.ndarray, aligned_activation: np.ndarray,
                                   prompt_text: str, response_base: str, response_aligned: str,
                                   layer_idx: int, prompt_idx: int, 
                                   sparsity_threshold: float = 0.1) -> None:
        """Create sparse visualizations showing sparsity patterns"""
        
        print(f"\n🔍 Creating sparse visualizations for prompt {prompt_idx}, layer {layer_idx}")
        print(f"   Sparsity threshold: {sparsity_threshold}")
        
        # Create binary masks for sparse activations
        base_sparse_mask = np.abs(base_activation) > sparsity_threshold
        aligned_sparse_mask = np.abs(aligned_activation) > sparsity_threshold
        
        # Calculate sparsity statistics
        base_sparsity = 1 - np.mean(base_sparse_mask)
        aligned_sparsity = 1 - np.mean(aligned_sparse_mask)
        
        print(f"   Base model sparsity: {base_sparsity:.3f}")
        print(f"   Aligned model sparsity: {aligned_sparsity:.3f}")
        
        # 1. Sparsity heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sparse Activations: Base vs Aligned (Layer {layer_idx}, Threshold={sparsity_threshold})', 
                    fontsize=16)
        
        # Base model sparsity pattern
        im1 = axes[0, 0].imshow(base_sparse_mask.T, aspect='auto', cmap='RdYlBu', interpolation='nearest')
        axes[0, 0].set_title(f'Base Model Sparse Pattern (Sparsity: {base_sparsity:.3f})')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Hidden Dimension')
        
        # Aligned model sparsity pattern
        im2 = axes[0, 1].imshow(aligned_sparse_mask.T, aspect='auto', cmap='RdYlBu', interpolation='nearest')
        axes[0, 1].set_title(f'Aligned Model Sparse Pattern (Sparsity: {aligned_sparsity:.3f})')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Hidden Dimension')
        
        # Difference in sparsity patterns
        sparsity_diff = aligned_sparse_mask.astype(int) - base_sparse_mask.astype(int)
        im3 = axes[1, 0].imshow(sparsity_diff.T, aspect='auto', cmap='RdBu', interpolation='nearest')
        axes[1, 0].set_title('Sparsity Pattern Difference (Aligned - Base)')
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Hidden Dimension')
        
        # Sparsity by position
        base_sparsity_by_pos = 1 - np.mean(base_sparse_mask, axis=1)
        aligned_sparsity_by_pos = 1 - np.mean(aligned_sparse_mask, axis=1)
        axes[1, 1].plot(base_sparsity_by_pos, label='Base', alpha=0.7, marker='o', markersize=3)
        axes[1, 1].plot(aligned_sparsity_by_pos, label='Aligned', alpha=0.7, marker='s', markersize=3)
        axes[1, 1].set_title('Sparsity by Sequence Position')
        axes[1, 1].set_xlabel('Sequence Position')
        axes[1, 1].set_ylabel('Sparsity (1 = fully sparse)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'sparse_activations_prompt_{prompt_idx}_layer_{layer_idx}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Sparsity statistics
        self._create_sparsity_statistics(base_activation, aligned_activation, 
                                       base_sparse_mask, aligned_sparse_mask,
                                       prompt_idx, layer_idx, sparsity_threshold)
    
    def _create_prompt_response_plot(self, prompt_text: str, response_base: str, response_aligned: str,
                                   prompt_idx: int, layer_idx: int) -> None:
        """Create a plot showing the prompt and responses"""
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(f'Prompt and Responses (Layer {layer_idx})', fontsize=16)
        
        # Prompt
        axes[0].text(0.05, 0.95, f"PROMPT:\n{prompt_text}", 
                    transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        axes[0].set_title('Input Prompt')
        axes[0].axis('off')
        
        # Base response
        axes[1].text(0.05, 0.95, f"BASE MODEL RESPONSE:\n{response_base}", 
                    transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        axes[1].set_title('Base Model Response')
        axes[1].axis('off')
        
        # Aligned response
        axes[2].text(0.05, 0.95, f"ALIGNED MODEL RESPONSE:\n{response_aligned}", 
                    transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
        axes[2].set_title('Aligned Model Response')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'prompt_responses_prompt_{prompt_idx}_layer_{layer_idx}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_sparsity_statistics(self, base_activation: np.ndarray, aligned_activation: np.ndarray,
                                  base_sparse_mask: np.ndarray, aligned_sparse_mask: np.ndarray,
                                  prompt_idx: int, layer_idx: int, sparsity_threshold: float) -> None:
        """Create detailed sparsity statistics plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Sparsity Analysis: Base vs Aligned (Layer {layer_idx})', fontsize=16)
        
        # 1. Sparsity distribution by dimension
        base_sparsity_by_dim = 1 - np.mean(base_sparse_mask, axis=0)
        aligned_sparsity_by_dim = 1 - np.mean(aligned_sparse_mask, axis=0)
        
        axes[0, 0].plot(base_sparsity_by_dim, label='Base', alpha=0.7)
        axes[0, 0].plot(aligned_sparsity_by_dim, label='Aligned', alpha=0.7)
        axes[0, 0].set_title('Sparsity by Hidden Dimension')
        axes[0, 0].set_xlabel('Hidden Dimension Index')
        axes[0, 0].set_ylabel('Sparsity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Activation magnitude distribution (sparse vs non-sparse)
        base_sparse_values = base_activation[base_sparse_mask]
        base_non_sparse_values = base_activation[~base_sparse_mask]
        aligned_sparse_values = aligned_activation[aligned_sparse_mask]
        aligned_non_sparse_values = aligned_activation[~aligned_sparse_mask]
        
        axes[0, 1].hist(base_sparse_values, bins=30, alpha=0.6, label='Base Sparse', density=True)
        axes[0, 1].hist(aligned_sparse_values, bins=30, alpha=0.6, label='Aligned Sparse', density=True)
        axes[0, 1].set_title('Activation Values (Sparse Elements)')
        axes[0, 1].set_xlabel('Activation Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sparsity vs activation magnitude
        base_magnitude = np.abs(base_activation.flatten())
        base_is_sparse = base_sparse_mask.flatten()
        aligned_magnitude = np.abs(aligned_activation.flatten())
        aligned_is_sparse = aligned_sparse_mask.flatten()
        
        # Subsample for visualization
        if len(base_magnitude) > 5000:
            indices = np.random.choice(len(base_magnitude), 5000, replace=False)
            base_mag_sample = base_magnitude[indices]
            base_sparse_sample = base_is_sparse[indices]
            aligned_mag_sample = aligned_magnitude[indices]
            aligned_sparse_sample = aligned_is_sparse[indices]
        else:
            base_mag_sample = base_magnitude
            base_sparse_sample = base_is_sparse
            aligned_mag_sample = aligned_magnitude
            aligned_sparse_sample = aligned_is_sparse
        
        axes[0, 2].scatter(base_mag_sample[base_sparse_sample], 
                          np.ones(np.sum(base_sparse_sample)) * 0.1, 
                          alpha=0.5, label='Base Sparse', s=1)
        axes[0, 2].scatter(base_mag_sample[~base_sparse_sample], 
                          np.ones(np.sum(~base_sparse_sample)) * 0.2, 
                          alpha=0.5, label='Base Non-Sparse', s=1)
        axes[0, 2].scatter(aligned_mag_sample[aligned_sparse_sample], 
                          np.ones(np.sum(aligned_sparse_sample)) * 0.3, 
                          alpha=0.5, label='Aligned Sparse', s=1)
        axes[0, 2].scatter(aligned_mag_sample[~aligned_sparse_sample], 
                          np.ones(np.sum(~aligned_sparse_sample)) * 0.4, 
                          alpha=0.5, label='Aligned Non-Sparse', s=1)
        axes[0, 2].set_title('Activation Magnitude vs Sparsity')
        axes[0, 2].set_xlabel('Activation Magnitude')
        axes[0, 2].set_ylabel('Model Type')
        axes[0, 2].set_yticks([0.1, 0.2, 0.3, 0.4])
        axes[0, 2].set_yticklabels(['Base Sparse', 'Base Non-Sparse', 'Aligned Sparse', 'Aligned Non-Sparse'])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Sparsity statistics summary
        stats_text = f"""
        SPARSITY STATISTICS (Threshold: {sparsity_threshold})
        
        Base Model:
        - Overall Sparsity: {1 - np.mean(base_sparse_mask):.3f}
        - Sparse Elements: {np.sum(base_sparse_mask):,}
        - Non-Sparse Elements: {np.sum(~base_sparse_mask):,}
        - Mean Sparse Value: {np.mean(base_sparse_values):.4f}
        - Mean Non-Sparse Value: {np.mean(base_non_sparse_values):.4f}
        
        Aligned Model:
        - Overall Sparsity: {1 - np.mean(aligned_sparse_mask):.3f}
        - Sparse Elements: {np.sum(aligned_sparse_mask):,}
        - Non-Sparse Elements: {np.sum(~aligned_sparse_mask):,}
        - Mean Sparse Value: {np.mean(aligned_sparse_values):.4f}
        - Mean Non-Sparse Value: {np.mean(aligned_non_sparse_values):.4f}
        """
        
        axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        axes[1, 0].set_title('Sparsity Statistics')
        axes[1, 0].axis('off')
        
        # 5. Sparsity pattern correlation
        sparsity_correlation = np.corrcoef(base_sparse_mask.flatten().astype(float), 
                                         aligned_sparse_mask.flatten().astype(float))[0, 1]
        axes[1, 1].text(0.5, 0.5, f'Sparsity Pattern\nCorrelation:\n{sparsity_correlation:.4f}', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        axes[1, 1].set_title('Sparsity Pattern Correlation')
        axes[1, 1].axis('off')
        
        # 6. Threshold sensitivity
        thresholds = np.logspace(-3, 0, 20)  # From 0.001 to 1.0
        base_sparsities = []
        aligned_sparsities = []
        
        for thresh in thresholds:
            base_mask = np.abs(base_activation) > thresh
            aligned_mask = np.abs(aligned_activation) > thresh
            base_sparsities.append(1 - np.mean(base_mask))
            aligned_sparsities.append(1 - np.mean(aligned_mask))
        
        axes[1, 2].semilogx(thresholds, base_sparsities, label='Base', marker='o', markersize=3)
        axes[1, 2].semilogx(thresholds, aligned_sparsities, label='Aligned', marker='s', markersize=3)
        axes[1, 2].axvline(sparsity_threshold, color='red', linestyle='--', alpha=0.7, label=f'Used Threshold: {sparsity_threshold}')
        axes[1, 2].set_title('Sparsity vs Threshold')
        axes[1, 2].set_xlabel('Threshold')
        axes[1, 2].set_ylabel('Sparsity')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'sparsity_analysis_prompt_{prompt_idx}_layer_{layer_idx}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_activations(self, model_family: str, prompt_idx: int, layer_idx: int, 
                            sparsity_threshold: float = 0.1, create_sparse: bool = True) -> None:
        """Main visualization function"""
        
        print(f"\n{'='*80}")
        print(f"VISUALIZING HIDDEN ACTIVATIONS")
        print(f"{'='*80}")
        print(f"Model Family: {model_family}")
        print(f"Prompt Index: {prompt_idx}")
        print(f"Layer Index: {layer_idx}")
        print(f"Sparsity Threshold: {sparsity_threshold}")
        print(f"Output Directory: {self.output_dir}")
        
        # Load data for both models
        try:
            base_data = self.load_model_data(f"{model_family}_base", 0)
            aligned_data = self.load_model_data(f"{model_family}_aligned", 1)
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return
        
        # Check if prompt index is valid
        if prompt_idx >= len(base_data['hidden_states']):
            print(f"❌ Prompt index {prompt_idx} out of range. Available: 0-{len(base_data['hidden_states'])-1}")
            return
        
        # Check if layer index is valid
        n_layers = base_data['hidden_states'].shape[1]
        if layer_idx < -n_layers or layer_idx >= n_layers:
            print(f"❌ Layer index {layer_idx} out of range. Available: {-n_layers} to {n_layers-1}")
            return
        
        # Extract activations
        base_activation, prompt_text, response_base = self.get_activation_for_prompt(
            base_data, prompt_idx, layer_idx)
        aligned_activation, _, response_aligned = self.get_activation_for_prompt(
            aligned_data, prompt_idx, layer_idx)
        
        print(f"\n📝 Prompt: {prompt_text[:100]}...")
        print(f"📊 Activation shapes: {base_activation.shape}")
        print(f"🔢 Layer {layer_idx} of {n_layers} layers")
        
        # Create regular visualizations
        self.create_regular_visualizations(
            base_activation, aligned_activation, prompt_text, 
            response_base, response_aligned, layer_idx, prompt_idx)
        
        # Create sparse visualizations if requested
        if create_sparse:
            self.create_sparse_visualizations(
                base_activation, aligned_activation, prompt_text,
                response_base, response_aligned, layer_idx, prompt_idx, sparsity_threshold)
        
        print(f"\n✅ Visualizations saved to: {self.output_dir}")
        print(f"   Files created:")
        print(f"   - regular_activations_prompt_{prompt_idx}_layer_{layer_idx}.png")
        print(f"   - distributions_prompt_{prompt_idx}_layer_{layer_idx}.png")
        print(f"   - prompt_responses_prompt_{prompt_idx}_layer_{layer_idx}.png")
        if create_sparse:
            print(f"   - sparse_activations_prompt_{prompt_idx}_layer_{layer_idx}.png")
            print(f"   - sparsity_analysis_prompt_{prompt_idx}_layer_{layer_idx}.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize hidden activations from LLaMA models')
    parser.add_argument('--model_family', type=str, default='llama2', 
                       help='Model family (llama2, falcon, mistral)')
    parser.add_argument('--prompt_idx', type=int, default=0, 
                       help='Index of prompt to visualize')
    parser.add_argument('--layer_idx', type=int, default=-1, 
                       help='Layer index to visualize (-1 for last layer)')
    parser.add_argument('--sparsity_threshold', type=float, default=0.1, 
                       help='Threshold for sparse visualization')
    parser.add_argument('--no_sparse', action='store_true', 
                       help='Skip sparse visualizations')
    parser.add_argument('--data_dir', type=str, default='./elicitation_data', 
                       help='Directory containing activation data')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ActivationVisualizer(data_dir=args.data_dir)
    
    # Create visualizations
    visualizer.visualize_activations(
        model_family=args.model_family,
        prompt_idx=args.prompt_idx,
        layer_idx=args.layer_idx,
        sparsity_threshold=args.sparsity_threshold,
        create_sparse=not args.no_sparse
    )

if __name__ == "__main__":
    main()


