# GenEval Environment

### For GPU below Hopper series

```bash
cd ULMEvalKit
mkdir -p ulmeval/dataset/geneval
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

conda create --name geneval python=3.8.10
conda activate geneval

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch==2.26.1
pip install clip-benchmark
pip install -U openmim
pip install einops
python -m pip install lightning
pip install tomli
pip install diffusers transformers
pip install platformdirs
pip install --upgrade setuptools

mim install mmengine mmcv-full==1.7.2

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

### For newer Hopper series

```bash
cd ULMEvalKit
mkdir -p ulmeval/dataset/geneval
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

conda create --name geneval python=3.10
conda activate geneval

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install open-clip-torch==2.26.1
pip install clip-benchmark
pip install -U openmim
pip install einops
python -m pip install lightning
pip install tomli
pip install diffusers transformers
pip install platformdirs
pip install --upgrade setuptools

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout 1.x
MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .
```



We refer to the environment installation guidance from [here](https://github.com/djghosh13/geneval/issues/12).
