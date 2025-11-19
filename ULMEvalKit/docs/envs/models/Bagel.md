# Bagel Environment

```bash
conda create -name bagel python=3.10 -y
conda activate bagel
git clone https://github.com/bytedance-seed/BAGEL.git
cd BAGEL
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```
