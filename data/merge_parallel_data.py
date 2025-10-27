#!/usr/bin/env python3
"""
Merge parallel data-split results into final datasets
"""

import h5py
import numpy as np
from pathlib import Path
import json
import shutil

def merge_model_data(model_key: str, total_splits: int = 6):
    """Merge data from all splits for a single model"""
    
    print(f"Merging data for {model_key}...")
    
    # Find all split files
    split_files = []
    for split_id in range(total_splits):
        split_file = Path(f"elicitation_data_{model_key}") / f"{model_key}_split_{split_id}_activations.h5"
        if split_file.exists():
            split_files.append(split_file)
        else:
            print(f"⚠️  Missing split file: {split_file}")
    
    if not split_files:
        print(f"❌ No split files found for {model_key}")
        return
    
    print(f"Found {len(split_files)} split files for {model_key}")
    
    # Create merged output file
    output_file = Path(f"elicitation_data_{model_key}") / f"{model_key}_activations.h5"
    
    with h5py.File(output_file, "w") as merged_hf:
        # Initialize datasets
        hidden_states_initialized = False
        all_prompt_ids = []
        all_responses = []
        all_domains = []
        all_variants = []
        
        for split_file in split_files:
            print(f"  Processing {split_file.name}...")
            
            with h5py.File(split_file, "r") as split_hf:
                # Initialize hidden_states dataset on first file
                if not hidden_states_initialized:
                    hidden_states_shape = split_hf["hidden_states"].shape
                    merged_hf.create_dataset(
                        "hidden_states",
                        shape=(0, *hidden_states_shape[1:]),
                        maxshape=(None, *hidden_states_shape[1:]),
                        dtype=split_hf["hidden_states"].dtype,
                        chunks=True,
                        compression="gzip"
                    )
                    hidden_states_initialized = True
                
                # Append hidden states
                hidden_data = split_hf["hidden_states"][:]
                curr_size = merged_hf["hidden_states"].shape[0]
                new_size = curr_size + hidden_data.shape[0]
                merged_hf["hidden_states"].resize(new_size, axis=0)
                merged_hf["hidden_states"][curr_size:new_size] = hidden_data
                
                # Collect metadata
                all_prompt_ids.extend(split_hf["prompt_ids"][:])
                all_responses.extend(split_hf["responses"][:])
                all_domains.extend(split_hf["domains"][:])
                all_variants.extend(split_hf["variants"][:])
        
        # Save metadata
        merged_hf.create_dataset("prompt_ids", data=np.array(all_prompt_ids, dtype=h5py.string_dtype()))
        merged_hf.create_dataset("responses", data=np.array(all_responses, dtype=h5py.string_dtype()))
        merged_hf.create_dataset("domains", data=np.array(all_domains, dtype=h5py.string_dtype()))
        merged_hf.create_dataset("variants", data=np.array(all_variants, dtype=np.int32))
    
    print(f"✅ Merged data saved to {output_file}")
    print(f"   Total samples: {len(all_prompt_ids)}")
    print(f"   Hidden states shape: {merged_hf['hidden_states'].shape}")
    
    return output_file

def merge_all_models():
    """Merge data for all models"""
    
    models = [
        "llama2_base",
        "llama2_aligned", 
        "falcon_base",
        "falcon_aligned",
        "mistral_base",
        "mistral_aligned"
    ]
    
    print("🔄 Merging parallel data-split results...")
    print("=" * 60)
    
    merged_files = []
    
    for model in models:
        try:
            merged_file = merge_model_data(model)
            if merged_file:
                merged_files.append(merged_file)
        except Exception as e:
            print(f"❌ Error merging {model}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 MERGING COMPLETE!")
    print(f"Successfully merged {len(merged_files)} models:")
    
    for file in merged_files:
        print(f"  - {file}")
    
    # Create final summary
    summary = {
        "models_processed": len(merged_files),
        "merged_files": [str(f) for f in merged_files],
        "total_samples_per_model": 9000,  # 1000 per domain * 3 domains * 3 variants
        "domains": ["logistics", "chemistry", "cyber"],
        "prompt_variants": 3
    }
    
    with open("parallel_merge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to: parallel_merge_summary.json")
    print("\n✅ Ready for geometric analysis!")

if __name__ == "__main__":
    merge_all_models()
