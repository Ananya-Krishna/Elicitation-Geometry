# Elicitation-Geometry

**Geometric Quantification of Elicitation Gaps via Scientific Machine Learning**

This repository contains a parallel data collection pipeline for extracting neural activations from large language models (LLMs) across multiple domains to study geometric properties of model behavior and refusal patterns.

## 🚀 Quick Start

### Environment Setup

1. **Create a conda environment:**
   ```bash
   conda create -n elicitation python=3.12
   conda activate elicitation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face authentication (for Llama-2 models):**
   ```bash
   # Generate a token at https://huggingface.co/settings/tokens
   # Save it to ~/.cache/huggingface/token
   echo "your_hf_token_here" > ~/.cache/huggingface/token
   ```

### Data Generation Pipeline

The pipeline generates synthetic data by extracting neural activations from 6 different LLMs across 3 domains (logistics, chemistry, cyber) with multiple prompt variants.

#### **Models Included:**
- **Llama-2-7b**: `meta-llama/Llama-2-7b-hf` (base) and `meta-llama/Llama-2-7b-chat-hf` (aligned)
- **Falcon-7b**: `tiiuae/falcon-7b` (base) and `tiiuae/falcon-7b-instruct` (aligned)  
- **Mistral-7B**: `mistralai/Mistral-7B-v0.1` (base) and `mistralai/Mistral-7B-Instruct-v0.2` (aligned)

#### **Data Domains:**
- **Logistics**: Traveling Salesman Problem (TSP) instances from OR-Library/TSPLIB
- **Chemistry**: Chemical reaction synthesis from USPTO dataset
- **Cyber**: Security challenges from picoCTF dataset

## 🏗️ Pipeline Architecture

### Directory Structure
```
Elicitation-Geometry/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
└── data/                          # All pipeline scripts
    ├── data_curation_h200_parallel_data.py  # Main parallel pipeline
    ├── data_integration.py                  # Real dataset prompt generation
    ├── data_loader.py                       # Data loading utilities
    ├── explore_data.py                      # Data exploration tool
    ├── merge_parallel_data.py               # Merge parallel results
    ├── run_parallel_*.slurm                 # 6 SLURM job scripts
    └── submit_parallel_jobs.sh              # Automated job submission
```

### Parallel Processing Design

The pipeline uses **data-splitting parallelization** for maximum efficiency:
- **6 parallel jobs**: Each processes 1 model with 1/6th of the data
- **True parallelization**: All models run simultaneously on different nodes
- **6x speedup**: ~1.5-2 hours vs 10+ hours sequential
- **158GB total data**: 9,000 samples × 6 models with full neural activations

## 🖥️ HPC Configuration (Bouchet)

### SLURM Configuration
The pipeline is optimized for Yale's Bouchet HPC system:

- **Partition**: `gpu_h200` (H200 GPUs)
- **Resources per job**: 1 GPU, 4 CPUs, 32GB RAM
- **Time limit**: 4 hours per job
- **Total resources**: 6 GPUs across multiple nodes

### Running the Pipeline

1. **Navigate to the data directory:**
   ```bash
   cd data/
   ```

2. **Submit all parallel jobs:**
   ```bash
   ./submit_parallel_jobs.sh
   ```

3. **Monitor progress:**
   ```bash
   squeue -u $USER
   tail -f data/logs/parallel_llama2_base_*.out
   ```

4. **Explore collected data:**
   ```bash
   python explore_data.py
   ```

5. **Merge results (optional):**
   ```bash
   python merge_parallel_data.py
   ```

## 📊 Output Data Structure

### Generated Files
Each model generates:
- **HDF5 file**: `{model}_split_{id}_activations.h5` (~13-50GB)
  - `hidden_states`: Neural activations (1500, 33, 512, 4096)
  - `responses`: Model-generated responses
  - `domains`: Domain labels (logistics/chemistry/cyber)
  - `variants`: Prompt variant IDs (0, 1, 2)
- **JSON file**: `prompts_split_{id}.json` (prompt metadata)

### Data Dimensions
- **Samples per model**: 1,500 (500 per domain × 3 variants)
- **Total samples**: 9,000 (6 models × 1,500)
- **Neural activations**: 33 layers × 512 tokens × 4096 dimensions
- **Total data size**: ~158GB

## 🔧 Customization

### Modifying Data Collection
Edit `data_curation_h200_parallel_data.py`:
- **Batch size**: `CONFIG["batch_size"]` (default: 32)
- **Sequence length**: `CONFIG["max_length"]` (default: 512)
- **Samples per domain**: `CONFIG["samples_per_domain"]` (default: 1000)
- **Prompt variants**: `CONFIG["prompt_variants"]` (default: 3)

### Adding New Models
1. Add model to `MODELS` dictionary in `data_curation_h200_parallel_data.py`
2. Create new SLURM script: `run_parallel_{model_name}.slurm`
3. Update `submit_parallel_jobs.sh`

### Adding New Domains
1. Extend `RealDataPromptGenerator` in `data_integration.py`
2. Add domain-specific prompt generation logic
3. Update domain list in configuration

## 📈 Performance Metrics

### Expected Runtime
- **Sequential processing**: ~10+ hours
- **Parallel processing**: ~1.5-2 hours
- **Speedup**: 6x improvement

### Resource Usage
- **GPU utilization**: 6 H200 GPUs (100% utilization)
- **Memory per job**: ~32GB RAM
- **Storage**: ~158GB total output

## 🐛 Troubleshooting

### Common Issues
1. **Authentication errors**: Ensure HF token is properly set
2. **GPU memory errors**: Reduce batch size in config
3. **Job timeouts**: Increase time limit in SLURM scripts
4. **Import errors**: Ensure all dependencies are installed

### Debugging
- Check job logs: `tail -f data/logs/parallel_*_*.out`
- Monitor GPU usage: `nvidia-smi`
- Check job status: `squeue -u $USER`

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{elicitation_geometry,
  title={Elicitation-Geometry: Parallel Data Collection Pipeline for Neural Activation Analysis},
  author={Krishna, Ananya and Kohli, Arjan},
  year={2024},
  url={https://github.com/Ananya-Krishna/Elicitation-Geometry}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the authors.