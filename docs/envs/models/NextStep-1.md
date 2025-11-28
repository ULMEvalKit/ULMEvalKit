# NextStep-1 Environment

```bash
conda create --name nextstep python=3.10
conda activate nextstep

# Clone and install NextStep-1
git clone https://github.com/stepfun-ai/NextStep-1.git
cd NextStep-1
pip install -e .

# Optional: Install flash-attention for better performance
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Download model weights from HuggingFace
pip install -U huggingface_hub
huggingface-cli download stepfun-ai/NextStep-1-f8ch16-Tokenizer
huggingface-cli download stepfun-ai/NextStep-1-Large
huggingface-cli download stepfun-ai/NextStep-1-Large-Edit
```

## Requirements
- Python: 3.10
- PyTorch: 2.5.1+cu121
- CUDA: 12.1
- cuDNN: 8.8.1.3
- Flash-Attention: 2.7.4.post1
- DeepSpeed: 0.16.3
- Transformers: 4.49.0

## Important Notes
- **VAE Tokenizer Path**: When using the VAE tokenizer, you **MUST** replace `stepfun-ai/NextStep-1-f8ch16-Tokenizer` with the absolute path to your downloaded tokenizer directory.
- **Torch Compile**: It is recommended to use `ENABLE_TORCH_COMPILE=false` to avoid compilation-related bugs.

## Troubleshooting
If you encounter Triton cache errors or torch.compile related issues:

**Solution 1**: Disable torch.compile
```bash
ENABLE_TORCH_COMPILE=false python inference.py
```

**Solution 2**: Install CUDA version of torch
```bash
pip install torch==2.5.1+cu121 torchvision triton --index-url https://download.pytorch.org/whl/cu121
```
