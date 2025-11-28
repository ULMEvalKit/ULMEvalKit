# Emu3 Environment

```bash
conda create --name emu3 python=3.10
conda activate emu3

# Clone and install Emu3
git clone https://github.com/baaivision/Emu3.git
cd Emu3
pip install -r requirements.txt
```

## Model Weights

| Model name               | HF Weight                                                      |
| ------------------------ | -------------------------------------------------------------- |
| **Emu3-Stage1**          | [🤗 HF link](https://huggingface.co/BAAI/Emu3-Stage1)          |
| **Emu3-Chat**            | [🤗 HF link](https://huggingface.co/BAAI/Emu3-Chat)            |
| **Emu3-Gen**             | [🤗 HF link](https://huggingface.co/BAAI/Emu3-Gen)             |
| **Emu3-VisionTokenizer** | [🤗 HF link](https://huggingface.co/BAAI/Emu3-VisionTokenizer) |

## Key Features
- **Emu3-Chat**: Specialized for vision-language understanding tasks
- **Emu3-Gen**: Specialized for vision generation tasks
- **Emu3-Stage1**: Image pretrained model supporting image captioning and 512x512 image generation
- **Emu3-VisionTokenizer**: Vision encoding and decoding component

## Requirements
- Python: 3.10+
- PyTorch with CUDA support
- Flash Attention 2 (recommended)
- Transformers library

## Usage Notes
- For image generation: Use Emu3-Gen model
- For vision-language understanding: Use Emu3-Chat model
- Supports flexible resolutions and styles for image generation
- Trained solely with next-token prediction without diffusion or compositional architectures
