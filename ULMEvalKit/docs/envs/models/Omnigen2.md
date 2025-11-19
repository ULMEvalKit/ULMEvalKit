# Omnigen2 Environment

```bash
conda create -name omnigen2 python=3.11
conda activate omnigen2
git clone git@github.com:VectorSpaceLab/OmniGen2.git
cd OmniGen2
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Note: Version 2.7.4.post1 is specified for compatibility with CUDA 12.4.
# Feel free to use a newer version if you use CUDA 12.6 or they fixed this compatibility issue.
# OmniGen2 runs even without flash-attn, though we recommend install it for best performance.
pip install flash-attn==2.7.4.post1 --no-build-isolation
```
