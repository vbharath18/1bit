# BitNet Model Inference and Benchmarking

This repository provides scripts and a Jupyter notebook for running and benchmarking the BitNet language model using Hugging Face Transformers. It is optimized for CPU inference and includes performance measurement utilities.

## Contents
- `bitnetv2.ipynb`: Jupyter notebook demonstrating BitNet model loading, inference, and performance optimizations.
- `bitnetv2.py`: Python script for benchmarking BitNet model inference speed and resource usage.
- `bitnet.py`: (Optional) Additional BitNet-related utilities or scripts.

## Setup
1. **Clone the repository** and navigate to the project directory.
2. **Create a virtual environment** (recommended):
   ```zsh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```zsh
   pip install torch git+https://github.com/huggingface/transformers.git bitsandbytes
   ```

## Usage
### Jupyter Notebook
Open `bitnetv2.ipynb` in Jupyter or VS Code to:
- Load the BitNet model
- Run inference on sample prompts
- Measure and display performance metrics

### Benchmark Script
Run the benchmark script from the command line:
```zsh
python bitnetv2.py -m microsoft/bitnet-b1.58-2B-4T -n 128 -p 512 -t 4
```
- `-m`: Model identifier or path
- `-n`: Number of tokens to generate
- `-p`: Prompt length
- `-t`: Number of CPU threads

## Notes
- The scripts are optimized for CPU inference. GPU is not required but will improve performance if available.
- For Apple Silicon (M1/M2/M3), the scripts default to CPU. MPS support is experimental in PyTorch/Transformers.
- Adjust batch size and thread count based on your system's resources.

## License
This project is licensed under the MIT License.
