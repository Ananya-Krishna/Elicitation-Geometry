#!/usr/bin/env python3
#!/usr/bin/env python3
"""
H200 Parallel Data-Split Pipeline (NumPy version)
- For each (MODEL_KEY, DATA_SPLIT), loads prompts_split_{split}.json
- Runs the model, extracts hidden states
- Saves activations as a single pooled NumPy file (.npy) per split

Output:
  ./numpy_activations/{MODEL_KEY}/{MODEL_KEY}_split_{DATA_SPLIT}_pooled.npy
  ./numpy_activations/{MODEL_KEY}/{MODEL_KEY}_split_{DATA_SPLIT}_metadata.npz
"""

from data_integration import RealDataPromptGenerator
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from tqdm import tqdm
import os
import gc
import sys

# Get script directory for path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ============================================================================
# PARALLEL DATA-SPLIT CONFIGURATION
# ============================================================================

# Get job parameters from command line
MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "llama2_base"
DATA_SPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TOTAL_SPLITS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

# Model mapping
MODELS = {
    "llama2_base": "meta-llama/Llama-2-7b-hf",
    "llama2_aligned": "meta-llama/Llama-2-7b-chat-hf",
    "falcon_base": "tiiuae/falcon-7b",
    "falcon_aligned": "tiiuae/falcon-7b-instruct",
    "mistral_base": "mistralai/Mistral-7B-v0.1",
    "mistral_aligned": "mistralai/Mistral-7B-Instruct-v0.2"
}

CONFIG = {
    "model_key": MODEL_KEY,
    "model_name": MODELS[MODEL_KEY],
    # IMPORTANT: new root so we don't overwrite old H5 dirs
    "output_dir": f"./numpy_activations/{MODEL_KEY}",
    "batch_size": 32,
    "max_length": 512,
    "samples_per_domain": 1000,
    "prompt_variants": 3,
    "data_split": DATA_SPLIT,
    "total_splits": TOTAL_SPLITS,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ============================================================================
# PARALLEL MODEL WRAPPER
# ============================================================================

class ParallelDataModelWrapper:
    """Model wrapper for parallel data processing"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        print(f"Loading {model_name} on H200...")
        
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=cache_dir,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                use_safetensors=True
            )
            
            if device != "cuda" or self.model.device.type != device:
                self.model = self.model.to(device)
                
            self.model.eval()
            self.device = device
            self.model_name = model_name
            
            print(f"✅ Successfully loaded {model_name} on H200")
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            raise
        
    def extract_hidden_states(self, prompts: list, max_length: int = 512) -> dict:
        """Extract hidden states for a batch of prompts"""
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # (num_layers, batch, seq_len, hidden_dim)
        hidden_states = torch.stack(outputs.hidden_states)
        
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    output_hidden_states=False,
                    use_cache=False,
                    past_key_values=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=1
                )
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
            generated = inputs["input_ids"]
        
        responses = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        return {
            "hidden_states": hidden_states.cpu().numpy(),      # (L, B, S, D)
            "attention_mask": inputs["attention_mask"].cpu().numpy(),  # (B, S)
            "input_ids": inputs["input_ids"].cpu().numpy(),    # (B, S)
            "responses": responses
        }

# ============================================================================
# PARALLEL DATA COLLECTOR
# ============================================================================

class ParallelDataCollector:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.prompt_gen = RealDataPromptGenerator(cache_dir="./dataset_cache")
    
    def generate_prompts_split(self) -> list:
        """Load prompts from existing split files or generate if not found"""
        print(f"Loading prompts for {self.config['model_key']} (split {self.config['data_split']}/{self.config['total_splits']})...")
        
        possible_roots = [
            PROJECT_ROOT,
            Path.cwd().resolve(),
            Path(self.output_dir).resolve().parent,
        ]
        
        existing_prompts_file = None
        for project_root in possible_roots:
            candidate = project_root / f"prompts_split_{self.config['data_split']}.json"
            if candidate.exists():
                existing_prompts_file = candidate
                break
        
        print(f"Looking for prompts file: {existing_prompts_file}")
        
        if existing_prompts_file and existing_prompts_file.exists():
            print(f"✓ Found existing prompts file: {existing_prompts_file}")
            with open(existing_prompts_file, 'r') as f:
                split_prompts = json.load(f)
            print(f"✓ Loaded {len(split_prompts)} prompts from existing file (RealToxicityPrompts)")
            
            # Copy to output directory for reference
            with open(self.output_dir / f"prompts_split_{self.config['data_split']}.json", "w") as f:
                json.dump(split_prompts, f, indent=2)
            
            return split_prompts
        
        # Fallback: Generate prompts if file doesn't exist
        print(f"⚠️  No existing prompts file found, generating new prompts...")
        print(f"Generating prompts for {self.config['model_key']} (split {self.config['data_split']}/{self.config['total_splits']})...")
        
        n_per_domain = self.config["samples_per_domain"]
        all_prompts = self.prompt_gen.generate_all_prompts(n_per_domain)
        
        # Create variants
        expanded_prompts = []
        for p in all_prompts:
            variants = self.prompt_gen.create_variants(p["prompt"], p["domain"])
            for v_idx, variant in enumerate(variants[:self.config["prompt_variants"]]):
                expanded_prompts.append({
                    "prompt": variant,
                    "domain": p["domain"],
                    "variant_id": v_idx,
                    "original_id": p["id"]
                })
        
        total_prompts = len(expanded_prompts)
        prompts_per_split = total_prompts // self.config["total_splits"]
        start_idx = self.config["data_split"] * prompts_per_split
        end_idx = start_idx + prompts_per_split if self.config["data_split"] < self.config["total_splits"] - 1 else total_prompts
        
        split_prompts = expanded_prompts[start_idx:end_idx]
        
        with open(self.output_dir / f"prompts_split_{self.config['data_split']}.json", "w") as f:
            json.dump(split_prompts, f, indent=2)
        
        print(f"✓ Generated {len(split_prompts)} prompts for split {self.config['data_split']}")
        return split_prompts
    
    def collect_model_data(self, prompts: list):
        """Collect pooled activations for the model with split data.

        - For each prompt, we compute a single hidden-dim pooled vector
          (final layer, mean over tokens using attention_mask).
        - We stream these to disk using a memmap, so memory stays small.
        """

        model_name = self.config["model_name"]
        model_key = self.config["model_key"]
        split_id = self.config["data_split"]
        batch_size = self.config["batch_size"]

        print(f"\n{'='*80}")
        print(f"Processing {model_key}: {model_name}")
        print(f"Data split: {split_id}/{self.config['total_splits']}")
        print(f"Prompts: {len(prompts)}")
        print(f"{'='*80}")

        if len(prompts) == 0:
            print("No prompts to process")
            return

        # -----------------------------
        # 1. Load model
        # -----------------------------
        model_wrapper = ParallelDataModelWrapper(model_name, self.config["device"])

        # -----------------------------
        # 2. Inspect a single prompt to infer hidden_dim
        # -----------------------------
        first_out = model_wrapper.extract_hidden_states(
            [prompts[0]["prompt"]],
            max_length=self.config["max_length"],
        )
        hidden_states_first = first_out["hidden_states"]  # (num_layers, 1, seq_len, hidden_dim)
        last_layer_first = hidden_states_first[-1, 0]     # (seq_len, hidden_dim)
        hidden_dim = last_layer_first.shape[-1]
        print(f"Detected hidden_dim = {hidden_dim}")

        # -----------------------------
        # 3. Create output files (memmap + metadata)
        # -----------------------------
        n_prompts = len(prompts)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # One pooled vector per prompt
        pooled_path = self.output_dir / f"{model_key}_split_{split_id}_pooled.npy"
        print(f"Creating memmap file: {pooled_path} with shape ({n_prompts}, {hidden_dim})")

        pooled_mm = np.memmap(
            pooled_path,
            dtype=np.float16,
            mode="w+",
            shape=(n_prompts, hidden_dim),
        )

        # We'll store small metadata in memory and save as NPZ at the end
        prompt_texts = []
        domains = []
        variants = []

        # -----------------------------
        # 4. Process prompts in batches
        # -----------------------------
        idx = 0
        for start in tqdm(
            range(0, n_prompts, batch_size),
            desc=f"Processing {model_key} split {split_id}"
        ):
            end = min(start + batch_size, n_prompts)
            batch_prompts = [p["prompt"] for p in prompts[start:end]]

            try:
                outputs = model_wrapper.extract_hidden_states(
                    batch_prompts,
                    max_length=self.config["max_length"],
                )
                hidden_states = outputs["hidden_states"]        # (num_layers, B, seq_len, hidden_dim)
                attn_mask = outputs["attention_mask"]           # (B, seq_len)

                # Use final layer: shape (B, seq_len, hidden_dim)
                last_layer = hidden_states[-1]  # (B, seq_len, hidden_dim)

                # Mean-pool over tokens using attention mask
                mask = attn_mask[..., None].astype(np.float32)  # (B, seq_len, 1)
                masked = last_layer * mask
                lengths = mask.sum(axis=1, keepdims=True)       # (B, 1, 1)
                lengths = np.clip(lengths, 1.0, None)           # avoid divide-by-zero
                pooled = masked.sum(axis=1) / lengths[:, 0, :]  # (B, hidden_dim)

                bsz = pooled.shape[0]
                pooled_mm[idx:idx + bsz] = pooled.astype(np.float16)
                idx += bsz

                # Metadata for this batch
                for p in prompts[start:end]:
                    prompt_texts.append(p["prompt"])
                    domains.append(p.get("domain", "unknown"))
                    variants.append(p.get("variant_id", -1))

            except Exception as e:
                print(f"Error processing batch {start}-{end}: {e}")
                continue

            # free big arrays early
            del hidden_states, attn_mask, last_layer, mask, masked, lengths, pooled
            gc.collect()

        # Flush memmap
        del pooled_mm
        gc.collect()

        print(f"Saved pooled activations to {pooled_path}")
        print(f"Total processed prompts: {idx}")

        # -----------------------------
        # 5. Save metadata
        # -----------------------------
        meta_path = self.output_dir / f"{model_key}_split_{split_id}_metadata.npz"
        np.savez(
            meta_path,
            prompts=np.array(prompt_texts, dtype=object),
            domains=np.array(domains, dtype=object),
            variants=np.array(variants, dtype=np.int32),
        )
        print(f"Saved metadata to {meta_path}")

        del model_wrapper
        gc.collect()
    
    def run_parallel_pipeline(self):
        """Run the parallel data collection pipeline"""
        print(f"Starting PARALLEL DATA-SPLIT pipeline for {self.config['model_key']}...")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.config['device']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Max length: {self.config['max_length']}")
        print(f"Data split: {self.config['data_split']}/{self.config['total_splits']}")
        
        prompts = self.generate_prompts_split()
        self.collect_model_data(prompts)
        
        print(f"\n🎉 PARALLEL DATA-SPLIT pipeline complete for {self.config['model_key']}!")
        print(f"All data saved to: {self.output_dir}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║   Warped Manifolds of Refusal - PARALLEL DATA-SPLIT Pipeline (NumPy) ║
    ║   Ananya Krishna & Arjan Kohli - S&DS 689                            ║
    ║   Model: {CONFIG['model_key']} (Split {CONFIG['data_split']}/{CONFIG['total_splits']}) ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    collector = ParallelDataCollector(CONFIG)
    collector.run_parallel_pipeline()
    
    print(f"\n📊 PARALLEL DATA-SPLIT Summary:")
    print(f"   - Model: {CONFIG['model_key']}")
    print(f"   - Data split: {CONFIG['data_split']}/{CONFIG['total_splits']}")
    print(f"   - Batch size: {CONFIG['batch_size']}")
    print(f"   - Max length: {CONFIG['max_length']}")
    print(f"\n✅ Ready for SAE training / geometric analysis!")
