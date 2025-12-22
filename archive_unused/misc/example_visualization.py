#!/usr/bin/env python3
"""
Example Usage of Hidden Activation Visualization Script

This script demonstrates how to use the visualization tools to explore
hidden activations from LLaMA models (base vs aligned).

Usage:
    python example_visualization.py
"""

import subprocess
import sys
from pathlib import Path

def run_visualization(model_family, prompt_idx, layer_idx, sparsity_threshold=0.1):
    """Run a single visualization"""
    cmd = [
        'python', 'visualize_activations_efficient.py',
        '--model_family', model_family,
        '--prompt_idx', str(prompt_idx),
        '--layer_idx', str(layer_idx),
        '--sparsity_threshold', str(sparsity_threshold)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Run example visualizations"""
    
    print("🔬 Hidden Activation Visualization Examples")
    print("=" * 60)
    
    # Check if the visualization script exists
    if not Path("visualize_activations_efficient.py").exists():
        print("❌ Error: visualize_activations_efficient.py not found!")
        print("   Make sure you're running this from the correct directory.")
        return
    
    # Example 1: Last layer of first prompt
    print("\n📊 Example 1: Last layer (layer -1) of first prompt")
    success1 = run_visualization("llama2", 0, -1, 0.1)
    
    # Example 2: Middle layer of a different prompt
    print("\n📊 Example 2: Middle layer (layer 16) of prompt 3")
    success2 = run_visualization("llama2", 3, 16, 0.05)
    
    # Example 3: First layer with different sparsity threshold
    print("\n📊 Example 3: First layer (layer 0) with high sparsity threshold")
    success3 = run_visualization("llama2", 2, 0, 0.2)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if success1 and success2 and success3:
        print("✅ All visualizations completed successfully!")
        print(f"📁 Check the 'activation_visualizations' directory for output files.")
    else:
        print("⚠️  Some visualizations failed. Check the error messages above.")
    
    print(f"\n📋 Generated files:")
    viz_dir = Path("activation_visualizations")
    if viz_dir.exists():
        for file in sorted(viz_dir.glob("*.png")):
            print(f"   - {file.name}")
    
    print(f"\n💡 Usage tips:")
    print(f"   - Use --help to see all options: python visualize_activations_efficient.py --help")
    print(f"   - Try different layers: --layer_idx 0 (first) to --layer_idx 32 (last)")
    print(f"   - Try different prompts: --prompt_idx 0 to --prompt_idx 1499")
    print(f"   - Adjust sparsity threshold: --sparsity_threshold 0.01 to 0.5")
    print(f"   - Skip sparse visualizations: --no_sparse")

if __name__ == "__main__":
    main()


