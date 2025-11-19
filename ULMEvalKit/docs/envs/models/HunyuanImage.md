# HunyuanImage Environment

```bash
conda create -name hunyuanimage python=3.10 -y
conda activate hunyuanimage
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git
cd HunyuanImage-3.0
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
pip install -r requirements.txt

# HunyuanImage runs even without flash-attn, though we recommend install it for best performance.
pip install flash-attn==2.8.3 --no-build-isolation
pip install flashinfer-python
```
