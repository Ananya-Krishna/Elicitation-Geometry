"""
Analyze Current Data and Build Training Pipeline
9,000 samples across 6 models
"""

import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# ============================================================================
# PART 1: DATA INVENTORY FOR YOUR ACTUAL PIPELINE
# ============================================================================

class DataInventory:
    """Analyze your actual parallel pipeline data"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
    
    def analyze_current_data(self) -> Dict:
        """Count and describe YOUR actual data"""
        
        print("="*80)
        print("ACTUAL DATA INVENTORY (PARALLEL PIPELINE)")
        print("="*80)
        
        inventory = {
            "prompts": {},
            "activations": {},
            "total_datapoints": {}
        }
        
        # 1. Analyze split prompt files
        prompt_files = list(self.data_dir.glob("prompts_split_*.json"))
        
        all_prompts = []
        for pf in prompt_files:
            with open(pf, 'r') as f:
                prompts = json.load(f)
                all_prompts.extend(prompts)
        
        if all_prompts:
            inventory["prompts"]["total"] = len(all_prompts)
            
            # Count by domain
            by_domain = {}
            by_variant = {}
            for p in all_prompts:
                domain = p.get("domain", "unknown")
                variant = p.get("variant_id", 0)
                by_domain[domain] = by_domain.get(domain, 0) + 1
                by_variant[variant] = by_variant.get(variant, 0) + 1
            
            inventory["prompts"]["by_domain"] = by_domain
            inventory["prompts"]["by_variant"] = by_variant
            
            print(f"\n📝 PROMPTS:")
            print(f"   Total unique prompts: {len(all_prompts):,}")
            print(f"   By domain: {by_domain}")
            print(f"   By variant: {by_variant}")
        
        # 2. Analyze H5 files (split format)
        h5_files = list(self.data_dir.glob("*_split_*_activations.h5"))
        
        print(f"\n💾 ACTIVATION FILES (.h5):")
        print(f"   Found {len(h5_files)} files")
        
        # Group by model family
        by_model = {}
        total_samples = 0
        
        for h5_file in sorted(h5_files):
            with h5py.File(h5_file, 'r') as hf:
                n_samples = hf["hidden_states"].shape[0]
                n_layers = hf["hidden_states"].shape[1]
                seq_len = hf["hidden_states"].shape[2]
                hidden_dim = hf["hidden_states"].shape[3]
                
                size_gb = h5_file.stat().st_size / (1024**3)
                
                # Extract model name (e.g., "llama2_base")
                name_parts = h5_file.stem.split("_split_")
                model_name = name_parts[0]
                split_id = name_parts[1].replace("_activations", "")
                
                if model_name not in by_model:
                    by_model[model_name] = {
                        "splits": [],
                        "total_samples": 0,
                        "shape": (n_layers, seq_len, hidden_dim)
                    }
                
                by_model[model_name]["splits"].append({
                    "split_id": split_id,
                    "samples": n_samples,
                    "size_gb": size_gb
                })
                by_model[model_name]["total_samples"] += n_samples
                total_samples += n_samples
        
        # Display by model
        for model_name, info in sorted(by_model.items()):
            print(f"\n   {model_name}:")
            print(f"     - Total samples: {info['total_samples']:,}")
            print(f"     - Shape per sample: {info['shape']}")
            for split in info["splits"]:
                print(f"     - Split {split['split_id']}: {split['samples']} samples, {split['size_gb']:.1f} GB")
        
        inventory["activations"] = by_model
        
        # 3. Calculate base-aligned pairs
        print(f"\n📊 DATA POINTS FOR TRAINING:")
        
        base_models = [k for k in by_model.keys() if "base" in k]
        aligned_models = [k for k in by_model.keys() if "aligned" in k]
        
        # Each base-aligned pair has same split ID
        model_families = set([m.replace("_base", "").replace("_aligned", "") for m in by_model.keys()])
        
        total_pairs = 0
        for family in model_families:
            base_key = f"{family}_base"
            aligned_key = f"{family}_aligned"
            if base_key in by_model and aligned_key in by_model:
                pairs = min(by_model[base_key]["total_samples"], 
                           by_model[aligned_key]["total_samples"])
                total_pairs += pairs
                print(f"   {family}: {pairs:,} pairs")
        
        print(f"\n   SUMMARY:")
        print(f"   ├─ Model families: {len(model_families)}")
        print(f"   ├─ Total prompts: {inventory['prompts'].get('total', 0):,}")
        print(f"   ├─ Total activations: {total_samples:,}")
        print(f"   └─ Base-aligned pairs: {total_pairs:,}")
        
        inventory["total_datapoints"] = {
            "prompts": inventory["prompts"].get("total", 0),
            "total_activations": total_samples,
            "base_aligned_pairs": total_pairs,
            "model_families": len(model_families)
        }
        
        # 4. Training splits
        print(f"\n🎯 RECOMMENDED SPLITS (per family, 80/10/10):")
        samples_per_family = total_pairs // len(model_families)
        print(f"   Per family: {samples_per_family:,} pairs")
        print(f"   Train: {int(samples_per_family * 0.8):,} pairs")
        print(f"   Val:   {int(samples_per_family * 0.1):,} pairs")
        print(f"   Test:  {int(samples_per_family * 0.1):,} pairs")
        
        print(f"\n   TOTAL ACROSS ALL FAMILIES:")
        print(f"   Train: {int(total_pairs * 0.8):,} pairs")
        print(f"   Val:   {int(total_pairs * 0.1):,} pairs")
        print(f"   Test:  {int(total_pairs * 0.1):,} pairs")
        
        print("\n" + "="*80)
        
        return inventory

# ============================================================================
# PART 2: PYTORCH DATASET FOR SPLIT DATA
# ============================================================================

class ParallelBaseAlignedDataset(Dataset):
    """Dataset for your parallel pipeline split data"""
    
    def __init__(self, 
                 data_dir: str,
                 model_family: str,
                 layer_idx: int = -1,
                 pool_method: str = "mean",
                 indices: Optional[List[int]] = None):
        """
        Args:
            data_dir: Directory with split H5 files
            model_family: "llama2", "falcon", or "mistral"
            layer_idx: Which layer to extract (-1 for last)
            pool_method: "mean", "max", or "last"
            indices: Optional list of sample indices (for train/val/test split)
        """
        self.data_dir = Path(data_dir)
        self.model_family = model_family
        self.layer_idx = layer_idx
        self.pool_method = pool_method
        
        # Find all split files for this model family
        base_files = sorted(self.data_dir.glob(f"{model_family}_base_split_*.h5"))
        aligned_files = sorted(self.data_dir.glob(f"{model_family}_aligned_split_*.h5"))
        
        if not base_files or not aligned_files:
            raise ValueError(f"No split files found for {model_family}")
        
        # Load all splits
        self.base_h5s = [h5py.File(f, 'r') for f in base_files]
        self.aligned_h5s = [h5py.File(f, 'r') for f in aligned_files]
        
        # Calculate cumulative sizes for indexing
        self.split_sizes = [h5["hidden_states"].shape[0] for h5 in self.base_h5s]
        self.cumulative_sizes = np.cumsum([0] + self.split_sizes)
        self.total_size = self.cumulative_sizes[-1]
        
        # Use subset if provided
        self.indices = indices if indices is not None else list(range(self.total_size))
        
        # Load metadata from first split
        self.domains = []
        self.variants = []
        for h5 in self.base_h5s:
            domains = [d.decode() if isinstance(d, bytes) else d for d in h5["domains"][:]]
            variants = h5["variants"][:]
            self.domains.extend(domains)
            self.variants.extend(variants.tolist())
        
        # Filter by indices
        self.domains = [self.domains[i] for i in self.indices]
        self.variants = [self.variants[i] for i in self.indices]
        
        # Detect refusals
        self.refusals = self._detect_refusals()
    
    def _get_split_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """Convert global index to (split_idx, local_idx)"""
        for split_idx in range(len(self.cumulative_sizes) - 1):
            if global_idx < self.cumulative_sizes[split_idx + 1]:
                local_idx = global_idx - self.cumulative_sizes[split_idx]
                return split_idx, local_idx
        raise IndexError(f"Index {global_idx} out of range")
    
    def _detect_refusals(self) -> np.ndarray:
        """Detect refusals in aligned model responses"""
        refusal_patterns = [
            "i cannot", "i can't", "i'm not able to", "i am not able to",
            "i cannot help", "i can't help", "i'm sorry, but",
            "not appropriate", "against my guidelines"
        ]
        
        refusals = []
        for idx in self.indices:
            split_idx, local_idx = self._get_split_and_local_idx(idx)
            response = self.aligned_h5s[split_idx]["responses"][local_idx]
            if isinstance(response, bytes):
                response = response.decode()
            response_lower = response.lower()
            is_refusal = any(pattern in response_lower for pattern in refusal_patterns)
            refusals.append(is_refusal)
        
        return np.array(refusals)
    
    def _pool_sequence(self, hidden_states: np.ndarray) -> np.ndarray:
        """Pool sequence dimension"""
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
        """Return a single base-aligned pair"""
        global_idx = self.indices[idx]
        split_idx, local_idx = self._get_split_and_local_idx(global_idx)
        
        # Get activations from correct split
        base_hidden = self.base_h5s[split_idx]["hidden_states"][local_idx]
        aligned_hidden = self.aligned_h5s[split_idx]["hidden_states"][local_idx]
        
        # Extract layer and pool
        base_layer = base_hidden[self.layer_idx]
        aligned_layer = aligned_hidden[self.layer_idx]
        
        base_pooled = self._pool_sequence(base_layer)
        aligned_pooled = self._pool_sequence(aligned_layer)
        
        return {
            "base": torch.FloatTensor(base_pooled),
            "aligned": torch.FloatTensor(aligned_pooled),
            "domain": self.domains[idx],
            "variant": self.variants[idx],
            "refusal": self.refusals[idx],
            "idx": global_idx
        }
    
    def close(self):
        """Close all H5 files"""
        for h5 in self.base_h5s:
            h5.close()
        for h5 in self.aligned_h5s:
            h5.close()

# ============================================================================
# PART 3: UPDATED TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Training pipeline for your 9,000 sample dataset"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def prepare_datasets(self, model_family: str = "llama2") -> Tuple[Dataset, Dataset, Dataset]:
        """Create train/val/test splits for your parallel data"""
        
        # Create full dataset
        full_dataset = ParallelBaseAlignedDataset(
            self.data_dir, 
            model_family,
            layer_idx=-1,
            pool_method="mean"
        )
        
        n_samples = len(full_dataset.indices)
        
        # Create splits
        indices = np.array(full_dataset.indices)
        train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        print(f"\n📊 Dataset splits for {model_family}:")
        print(f"   Total samples: {n_samples:,}")
        print(f"   Train: {len(train_idx):,} ({len(train_idx)/n_samples*100:.1f}%)")
        print(f"   Val:   {len(val_idx):,} ({len(val_idx)/n_samples*100:.1f}%)")
        print(f"   Test:  {len(test_idx):,} ({len(test_idx)/n_samples*100:.1f}%)")
        
        # Create datasets with split indices
        train_dataset = ParallelBaseAlignedDataset(
            self.data_dir, model_family, indices=train_idx.tolist()
        )
        val_dataset = ParallelBaseAlignedDataset(
            self.data_dir, model_family, indices=val_idx.tolist()
        )
        test_dataset = ParallelBaseAlignedDataset(
            self.data_dir, model_family, indices=test_idx.tolist()
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_refusal_statistics(self, dataset: Dataset) -> Dict:
        """Analyze refusal patterns"""
        refusals = np.array([dataset[i]["refusal"] for i in range(len(dataset))])
        domains = [dataset[i]["domain"] for i in range(len(dataset))]
        
        stats = {
            "total_refusals": refusals.sum(),
            "refusal_rate": refusals.mean(),
            "by_domain": {}
        }
        
        for domain in set(domains):
            domain_mask = np.array([d == domain for d in domains])
            domain_refusals = refusals[domain_mask]
            stats["by_domain"][domain] = {
                "count": domain_refusals.sum(),
                "rate": domain_refusals.mean()
            }
        
        return stats

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Analyze YOUR actual data"""
    
    # Step 1: Inventory
    print("\n" + "="*80)
    print("ANALYZING YOUR PARALLEL PIPELINE DATA")
    print("="*80 + "\n")
    
    inventory = DataInventory("./data")
    summary = inventory.analyze_current_data()
    
    # Step 2: Load datasets
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    
    pipeline = TrainingPipeline("./data")
    
    try:
        train_ds, val_ds, test_ds = pipeline.prepare_datasets("llama2")
        
        # Step 3: Analyze refusals
        print("\n" + "="*80)
        print("REFUSAL ANALYSIS")
        print("="*80)
        
        refusal_stats = pipeline.get_refusal_statistics(train_ds)
        print(f"\n   Overall refusal rate: {refusal_stats['refusal_rate']:.1%}")
        print(f"   Total refusals: {refusal_stats['total_refusals']}")
        print("\n   By domain:")
        for domain, stats in refusal_stats["by_domain"].items():
            print(f"     {domain}: {stats['rate']:.1%} ({stats['count']} refusals)")
        
        # Clean up
        train_ds.close()
        val_ds.close()
        test_ds.close()
        
        print("\n✅ Your data is ready for training!")
        print(f"\n📊 Key numbers:")
        print(f"   - Total pairs: {summary['total_datapoints']['base_aligned_pairs']:,}")
        print(f"   - Training pairs: ~{int(summary['total_datapoints']['base_aligned_pairs'] * 0.8):,}")
        print(f"   - This is SUFFICIENT for your geometric analysis!")
        
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("Make sure your data files are in ./data/ directory")

if __name__ == "__main__":
    main()