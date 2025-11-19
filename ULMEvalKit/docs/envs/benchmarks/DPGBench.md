# DPGBench Environment


```bash
conda create --name dpgbench python=3.10
conda activate dpgbench
pip install accelerate
pip install numpy
pip install pandas
pip install pillow
pip install tqdm

# for modelscope
pip install cloudpickle
pip install decord>=0.6.0
pip install diffusers
pip install fairseq
pip install ftfy>=6.0.3
pip install librosa==0.10.1
pip install modelscope
pip install opencv-python
# compatible with taming-transformers-rom1504
pip install rapidfuzz
# rough-score was just recently updated from 0.0.4 to 0.0.7
# which introduced compatability issues that are being investigated
pip install rouge_score<=0.0.4
pip install safetensors
# scikit-video
pip install soundfile
pip install taming-transformers-rom1504
pip install tiktoken
pip install timm
pip install tokenizers
pip install torchvision
pip install transformers
pip install transformers_stream_generator
pip install unicodedata2
pip install zhconv
```
