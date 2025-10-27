#!/usr/bin/env python3
"""
Data Structure Explorer
Shows how the collected data is organized
"""

import h5py
import json
import numpy as np
from pathlib import Path

def explore_model_data(model_name, split_id):
    """Explore the data structure for a specific model"""
    
    print(f"\n{'='*80}")
    print(f"EXPLORING DATA FOR: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Paths
    data_dir = Path(f"elicitation_data_{model_name}")
    h5_file = data_dir / f"{model_name}_split_{split_id}_activations.h5"
    prompts_file = data_dir / f"prompts_split_{split_id}.json"
    
    print(f"📁 Data Directory: {data_dir}")
    print(f"📊 HDF5 File: {h5_file}")
    print(f"📝 Prompts File: {prompts_file}")
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    print(f"\n📋 PROMPTS DATA:")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Sample prompt: {prompts[0]['prompt'][:100]}...")
    print(f"   Domain: {prompts[0]['domain']}")
    print(f"   Variant: {prompts[0]['variant_id']}")
    
    # Load HDF5 data
    with h5py.File(h5_file, 'r') as f:
        print(f"\n🧠 NEURAL ACTIVATIONS (HDF5):")
        print(f"   Hidden states shape: {f['hidden_states'].shape}")
        print(f"   Number of layers: {f['hidden_states'].shape[1]}")
        print(f"   Sequence length: {f['hidden_states'].shape[2]}")
        print(f"   Hidden dimension: {f['hidden_states'].shape[3]}")
        
        print(f"\n💬 MODEL RESPONSES:")
        print(f"   Number of responses: {len(f['responses'])}")
        print(f"   Sample response 1: {f['responses'][0].decode('utf-8')[:200]}...")
        print(f"   Sample response 2: {f['responses'][1].decode('utf-8')[:200]}...")
        print(f"   Sample response 3: {f['responses'][2].decode('utf-8')[:200]}...")
        
        print(f"\n🏷️  METADATA:")
        print(f"   Domains: {set(f['domains'][:].astype(str))}")
        print(f"   Variants: {set(f['variants'][:])}")
        print(f"   Prompt IDs: {f['prompt_ids'][0].decode('utf-8')} (sample)")
        
        # Show complete prompt-response pairs
        print(f"\n📖 COMPLETE PROMPT-RESPONSE PAIRS:")
        for i in range(min(3, len(prompts))):
            print(f"\n--- PAIR {i+1} ---")
            print(f"PROMPT: {prompts[i]['prompt']}")
            print(f"RESPONSE: {f['responses'][i].decode('utf-8')}")
            print(f"DOMAIN: {f['domains'][i].decode('utf-8')}")
            print(f"VARIANT: {f['variants'][i]}")

def show_data_summary():
    """Show summary of all collected data"""
    
    print(f"\n{'='*80}")
    print("COMPLETE DATA COLLECTION SUMMARY")
    print(f"{'='*80}")
    
    models = [
        ("llama2_base", 0),
        ("llama2_aligned", 1), 
        ("falcon_base", 2),
        ("falcon_aligned", 3),
        ("mistral_base", 4),
        ("mistral_aligned", 5)
    ]
    
    total_samples = 0
    total_size = 0
    
    for model_name, split_id in models:
        data_dir = Path(f"elicitation_data_{model_name}")
        h5_file = data_dir / f"{model_name}_split_{split_id}_activations.h5"
        
        if h5_file.exists():
            file_size = h5_file.stat().st_size / (1024**3)  # GB
            total_size += file_size
            
            with h5py.File(h5_file, 'r') as f:
                samples = f['hidden_states'].shape[0]
                total_samples += samples
                
                print(f"✅ {model_name:15} | {samples:4d} samples | {file_size:6.1f} GB")
        else:
            print(f"❌ {model_name:15} | Not found")
    
    print(f"\n📊 TOTALS:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Total data size: {total_size:.1f} GB")
    print(f"   Average per model: {total_samples/len(models):.0f} samples")

if __name__ == "__main__":
    # Show summary first
    show_data_summary()
    
    # Explore one model in detail
    explore_model_data("llama2_base", 0)
