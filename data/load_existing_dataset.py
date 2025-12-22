"""
MINIMAL WORKING VERSION - Load RealToxicityPrompts
Save this as: data/load_existing_dataset.py
"""

from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

print("Loading dataset...")
dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
print(f"Loaded {len(dataset)} prompts")

# Filter cyber content
def filter_cyber(example):
    text = example['prompt']['text'].lower()
    cyber_keywords = [
        'hack', 'hacker', 'exploit', 'vulnerability', 'backdoor',
        'malware', 'virus', 'trojan', 'ransomware', 'keylogger',
        'ddos', 'phishing', 'sql injection', 'xss', 'buffer overflow',
        'password crack', 'brute force', 'nmap', 'metasploit',
        'data breach', 'stolen credential'
    ]
    return not any(kw in text for kw in cyber_keywords)

print("Filtering cyber content...")
dataset = dataset.filter(filter_cyber)
print(f"After filtering: {len(dataset)} prompts")

# Stratify by toxicity
print("\nStratifying by toxicity...")
buckets = [
    (0.0, 0.25, 'benign'),
    (0.25, 0.5, 'borderline'),
    (0.5, 0.75, 'concerning'),
    (0.75, 1.0, 'toxic')
]

all_samples = []
samples_per_bucket = 9000 // 4  # 2250 per bucket

for low, high, label in buckets:
    print(f"  Sampling {label} [{low}, {high})...")
    
    # Collect all examples in this range
    bucket_data = []
    for example in tqdm(dataset):
        toxicity = example['prompt']['toxicity']
        if toxicity is not None and low <= toxicity < high:
            bucket_data.append(example)
    
    print(f"    Found {len(bucket_data)} prompts")
    
    # Sample from bucket
    if len(bucket_data) >= samples_per_bucket:
        indices = np.random.choice(len(bucket_data), samples_per_bucket, replace=False)
        selected = [bucket_data[i] for i in indices]
    else:
        print(f"    Warning: Using all {len(bucket_data)} available")
        selected = bucket_data
    
    all_samples.extend(selected)

print(f"\nTotal samples: {len(all_samples)}")

# Split into 6 files for parallel processing
print("\nCreating 6 split files...")
samples_per_split = len(all_samples) // 6

for split_id in range(6):
    start = split_id * samples_per_split
    end = start + samples_per_split if split_id < 5 else len(all_samples)
    
    split_samples = all_samples[start:end]
    
    # Format for your pipeline
    formatted = []
    for idx, sample in enumerate(split_samples):
        toxicity = sample['prompt']['toxicity']
        
        # Assign domain based on toxicity
        if toxicity < 0.25:
            domain = "toxic_benign"
        elif toxicity < 0.5:
            domain = "toxic_borderline"
        elif toxicity < 0.75:
            domain = "toxic_concerning"
        else:
            domain = "toxic_high"
        
        formatted.append({
            "id": f"realtox_{idx}_s{split_id}",
            "domain": domain,
            "prompt": sample['prompt']['text'],
            "variant_id": 0,
            "toxicity": float(toxicity),
            "source": "RealToxicityPrompts"
        })
    
    # Save
    output_file = f"prompts_split_{split_id}.json"
    with open(output_file, 'w') as f:
        json.dump(formatted, f, indent=2)
    
    print(f"  ✓ {output_file}: {len(formatted)} prompts")

print("\n✅ Done! Ready to run: ./submit_parallel_jobs.sh")