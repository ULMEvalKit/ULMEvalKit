# HiDream Environment

```bash
conda create -name hidream python=3.10 -y
conda activate hidream
git clone https://github.com/HiDream-ai/HiDream-I1.git
cd HiDream-I1
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
pip install git+https://github.com/huggingface/diffusers.git
```
