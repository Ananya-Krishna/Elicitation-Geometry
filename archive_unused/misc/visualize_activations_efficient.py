#!/usr/bin/env python3
"""
Memory-Efficient Hidden Activation Visualization Script
Visualizes hidden activations from LLaMA models (base vs aligned) for a single prompt

This script is optimized for large datasets and loads only the necessary data.

Usage:
    python visualize_activations_efficient.py --model_family llama2 --prompt_idx 0 --layer_idx -1
    python visualize_activations_efficient.py --model_family llama2 --prompt_idx 5 --layer_idx 10 --sparse_threshold 0.1
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class EfficientActivationVisualizer:
    """Memory-efficient visualizer for hidden activations from LLaMA models"""
    
    def __init__(self, data_dir: str = "./elicitation_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("./activation_visualizations")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_single_activation(self, model_name: str, split_id: int, prompt_idx: int, layer_idx: int) -> Tuple[np.ndarray, str, str]:
        """Load only the specific activation we need"""
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
        
        # Load only the specific activation we need
        with h5py.File(h5_file, 'r') as f:
            # Load only the specific sample and layer
            hidden_states = f['hidden_states'][prompt_idx, layer_idx, :, :]  # [seq_len, hidden_dim]
            response = f['responses'][prompt_idx].decode() if isinstance(f['responses'][prompt_idx], bytes) else f['responses'][prompt_idx]
        
        prompt_text = prompts[prompt_idx]['prompt']
        
        return hidden_states, prompt_text, response
    
    def create_regular_visualizations(self, base_activation: np.ndarray, aligned_activation: np.ndarray,
                                    prompt_text: str, response_base: str, response_aligned: str,
                                    layer_idx: int, prompt_idx: int) -> None:
        """Create regular (non-sparse) visualizations"""
        
        print(f"\n📊 Creating regular visualizations for prompt {prompt_idx}, layer {layer_idx}")
        
        # 1. Activation heatmaps (subsample for memory efficiency)
        seq_len, hidden_dim = base_activation.shape
        
        # Subsample hidden dimensions for visualization
        if hidden_dim > 1000:
            dim_indices = np.linspace(0, hidden_dim-1, 1000, dtype=int)
            base_viz = base_activation[:, dim_indices]
            aligned_viz = aligned_activation[:, dim_indices]
        else:
            base_viz = base_activation
            aligned_viz = aligned_activation
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Hidden Activations: Base vs Aligned (Layer {layer_idx})', fontsize=16)
        
        # Base model heatmap
        im1 = axes[0, 0].imshow(base_viz.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[0, 0].set_title('Base Model Activations')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Hidden Dimension (subsampled)')
        plt.colorbar(im1, ax=axes[0, 0], label='Activation Value')
        
        # Aligned model heatmap
        im2 = axes[0, 1].imshow(aligned_viz.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[0, 1].set_title('Aligned Model Activations')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Hidden Dimension (subsampled)')
        plt.colorbar(im2, ax=axes[0, 1], label='Activation Value')
        
        # Difference heatmap
        diff_activation = aligned_viz - base_viz
        im3 = axes[1, 0].imshow(diff_activation.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        axes[1, 0].set_title('Difference (Aligned - Base)')
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Hidden Dimension (subsampled)')
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
        
        # 2. Distribution plots (subsample for memory efficiency)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Activation Distributions: Base vs Aligned (Layer {layer_idx})', fontsize=16)
        
        # Subsample for distribution analysis
        if base_activation.size > 100000:
            # Randomly sample 100k elements
            flat_indices = np.random.choice(base_activation.size, 100000, replace=False)
            base_flat = base_activation.flatten()[flat_indices]
            aligned_flat = aligned_activation.flatten()[flat_indices]
        else:
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
        
        # Scatter plot: base vs aligned (further subsample)
        if len(base_flat) > 10000:
            scatter_indices = np.random.choice(len(base_flat), 10000, replace=False)
            base_sample = base_flat[scatter_indices]
            aligned_sample = aligned_flat[scatter_indices]
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
        
        # 1. Sparsity heatmaps (subsample for memory efficiency)
        seq_len, hidden_dim = base_activation.shape
        
        # Subsample hidden dimensions for visualization
        if hidden_dim > 1000:
            dim_indices = np.linspace(0, hidden_dim-1, 1000, dtype=int)
            base_sparse_viz = base_sparse_mask[:, dim_indices]
            aligned_sparse_viz = aligned_sparse_mask[:, dim_indices]
        else:
            base_sparse_viz = base_sparse_mask
            aligned_sparse_viz = aligned_sparse_mask
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sparse Activations: Base vs Aligned (Layer {layer_idx}, Threshold={sparsity_threshold})', 
                    fontsize=16)
        
        # Base model sparsity pattern
        im1 = axes[0, 0].imshow(base_sparse_viz.T, aspect='auto', cmap='RdYlBu', interpolation='nearest')
        axes[0, 0].set_title(f'Base Model Sparse Pattern (Sparsity: {base_sparsity:.3f})')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Hidden Dimension (subsampled)')
        
        # Aligned model sparsity pattern
        im2 = axes[0, 1].imshow(aligned_sparse_viz.T, aspect='auto', cmap='RdYlBu', interpolation='nearest')
        axes[0, 1].set_title(f'Aligned Model Sparse Pattern (Sparsity: {aligned_sparsity:.3f})')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Hidden Dimension (subsampled)')
        
        # Difference in sparsity patterns
        sparsity_diff = aligned_sparse_viz.astype(int) - base_sparse_viz.astype(int)
        im3 = axes[1, 0].imshow(sparsity_diff.T, aspect='auto', cmap='RdBu', interpolation='nearest')
        axes[1, 0].set_title('Sparsity Pattern Difference (Aligned - Base)')
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Hidden Dimension (subsampled)')
        
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
        
        # 1. Sparsity distribution by dimension (subsample for memory)
        base_sparsity_by_dim = 1 - np.mean(base_sparse_mask, axis=0)
        aligned_sparsity_by_dim = 1 - np.mean(aligned_sparse_mask, axis=0)
        
        # Subsample dimensions for visualization
        if len(base_sparsity_by_dim) > 1000:
            dim_indices = np.linspace(0, len(base_sparsity_by_dim)-1, 1000, dtype=int)
            base_sparsity_viz = base_sparsity_by_dim[dim_indices]
            aligned_sparsity_viz = aligned_sparsity_by_dim[dim_indices]
        else:
            base_sparsity_viz = base_sparsity_by_dim
            aligned_sparsity_viz = aligned_sparsity_by_dim
        
        axes[0, 0].plot(base_sparsity_viz, label='Base', alpha=0.7)
        axes[0, 0].plot(aligned_sparsity_viz, label='Aligned', alpha=0.7)
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
        
        # Subsample for visualization
        max_samples = 50000
        if len(base_sparse_values) > max_samples:
            base_sparse_sample = np.random.choice(base_sparse_values, max_samples, replace=False)
        else:
            base_sparse_sample = base_sparse_values
            
        if len(aligned_sparse_values) > max_samples:
            aligned_sparse_sample = np.random.choice(aligned_sparse_values, max_samples, replace=False)
        else:
            aligned_sparse_sample = aligned_sparse_values
        
        axes[0, 1].hist(base_sparse_sample, bins=30, alpha=0.6, label='Base Sparse', density=True)
        axes[0, 1].hist(aligned_sparse_sample, bins=30, alpha=0.6, label='Aligned Sparse', density=True)
        axes[0, 1].set_title('Activation Values (Sparse Elements)')
        axes[0, 1].set_xlabel('Activation Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sparsity vs activation magnitude (subsample heavily)
        base_magnitude = np.abs(base_activation.flatten())
        base_is_sparse = base_sparse_mask.flatten()
        aligned_magnitude = np.abs(aligned_activation.flatten())
        aligned_is_sparse = aligned_sparse_mask.flatten()
        
        # Subsample for visualization
        if len(base_magnitude) > 10000:
            indices = np.random.choice(len(base_magnitude), 10000, replace=False)
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
        
        # 6. Threshold sensitivity (subsample for speed)
        thresholds = np.logspace(-3, 0, 10)  # Reduced from 20 to 10
        base_sparsities = []
        aligned_sparsities = []
        
        # Subsample activations for threshold analysis
        if base_activation.size > 100000:
            sample_indices = np.random.choice(base_activation.size, 100000, replace=False)
            base_sample = base_activation.flatten()[sample_indices]
            aligned_sample = aligned_activation.flatten()[sample_indices]
        else:
            base_sample = base_activation.flatten()
            aligned_sample = aligned_activation.flatten()
        
        for thresh in thresholds:
            base_mask = np.abs(base_sample) > thresh
            aligned_mask = np.abs(aligned_sample) > thresh
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
            base_activation, prompt_text, response_base = self.load_single_activation(
                f"{model_family}_base", 0, prompt_idx, layer_idx)
            aligned_activation, _, response_aligned = self.load_single_activation(
                f"{model_family}_aligned", 1, prompt_idx, layer_idx)
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return
        
        print(f"\n📝 Prompt: {prompt_text[:100]}...")
        print(f"📊 Activation shapes: {base_activation.shape}")
        
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
    visualizer = EfficientActivationVisualizer(data_dir=args.data_dir)
    
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


