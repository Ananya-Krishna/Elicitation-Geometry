#!/usr/bin/env python3
"""
H200 Parallel Data-Split Pipeline
Splits data across multiple jobs for true parallelization
"""

from data_integration import RealDataPromptGenerator
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import h5py
from pathlib import Path
from tqdm import tqdm
import os
import psutil
import gc
import sys

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
    "output_dir": f"./elicitation_data_{MODEL_KEY}",
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
            padding_side='left'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
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
            "hidden_states": hidden_states.cpu().numpy(),
            "attention_mask": inputs["attention_mask"].cpu().numpy(),
            "input_ids": inputs["input_ids"].cpu().numpy(),
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
        """Generate prompts and split them across jobs"""
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
        
        # Split data across jobs
        total_prompts = len(expanded_prompts)
        prompts_per_split = total_prompts // self.config["total_splits"]
        start_idx = self.config["data_split"] * prompts_per_split
        end_idx = start_idx + prompts_per_split if self.config["data_split"] < self.config["total_splits"] - 1 else total_prompts
        
        split_prompts = expanded_prompts[start_idx:end_idx]
        
        # Save split prompts
        with open(self.output_dir / f"prompts_split_{self.config['data_split']}.json", "w") as f:
            json.dump(split_prompts, f, indent=2)
        
        print(f"✓ Generated {len(split_prompts)} prompts for split {self.config['data_split']}")
        return split_prompts
    
    def collect_model_data(self, prompts: list):
        """Collect data for the model with split data"""
        model_name = self.config["model_name"]
        print(f"\n{'='*80}")
        print(f"Processing {self.config['model_key']}: {model_name} on H200")
        print(f"Data split: {self.config['data_split']}/{self.config['total_splits']}")
        print(f"Prompts: {len(prompts)}")
        print(f"{'='*80}")
        
        model_wrapper = ParallelDataModelWrapper(model_name, self.config["device"])
        output_file = self.output_dir / f"{self.config['model_key']}_split_{self.config['data_split']}_activations.h5"
        batch_size = self.config["batch_size"]
        
        with h5py.File(output_file, "w") as hf:
            if len(prompts) == 0:
                print("No prompts to process")
                return
                
            first_batch = [prompts[0]["prompt"]]
            first_output = model_wrapper.extract_hidden_states(first_batch)
            
            num_layers, _, _, hidden_dim = first_output["hidden_states"].shape
            
            hf.create_dataset(
                "hidden_states",
                shape=(0, num_layers, self.config["max_length"], hidden_dim),
                maxshape=(None, num_layers, self.config["max_length"], hidden_dim),
                dtype=np.float16,
                chunks=True,
                compression="gzip"
            )
            
            responses = []
            prompt_ids = []
            
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {self.config['model_key']} split {self.config['data_split']}"):
                batch_prompts = [p["prompt"] for p in prompts[i:i+batch_size]]
                batch_ids = [f"split_{self.config['data_split']}_batch_{i//batch_size}_v{p['variant_id']}" for p in prompts[i:i+batch_size]]
                
                try:
                    outputs = model_wrapper.extract_hidden_states(
                        batch_prompts,
                        max_length=self.config["max_length"]
                    )
                    
                    hidden = outputs["hidden_states"]
                    hidden = np.transpose(hidden, (1, 0, 2, 3))
                    
                    curr_seq_len = hidden.shape[2]
                    if curr_seq_len < self.config["max_length"]:
                        pad_width = ((0, 0), (0, 0), (0, self.config["max_length"] - curr_seq_len), (0, 0))
                        hidden = np.pad(hidden, pad_width, mode='constant', constant_values=0)
                    else:
                        hidden = hidden[:, :, :self.config["max_length"], :]
                    
                    curr_size = hf["hidden_states"].shape[0]
                    new_size = curr_size + hidden.shape[0]
                    hf["hidden_states"].resize(new_size, axis=0)
                    hf["hidden_states"][curr_size:new_size] = hidden.astype(np.float16)
                    
                    responses.extend(outputs["responses"])
                    prompt_ids.extend(batch_ids)
                    
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue
            
            hf.create_dataset("prompt_ids", data=np.array(prompt_ids, dtype=h5py.string_dtype()))
            hf.create_dataset("responses", data=np.array(responses, dtype=h5py.string_dtype()))
            
            domains = [p["domain"] for p in prompts]
            variants = [p["variant_id"] for p in prompts]
            hf.create_dataset("domains", data=np.array(domains, dtype=h5py.string_dtype()))
            hf.create_dataset("variants", data=np.array(variants, dtype=np.int32))
        
        print(f"Saved activations to {output_file}")
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
    ║   Warped Manifolds of Refusal - PARALLEL DATA-SPLIT Pipeline        ║
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
    print(f"\n✅ Ready for geometric analysis!")
