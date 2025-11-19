# T2I-CompBench Guide

## Reference Rquirements
```bash
conda create -name t2icompbench python=3.10.0
conda activate t2icompbench
pip install -r ulmeval/dataset/utils/t2i_compbench/requirements.txt
```

## Download detection model weights
```bash
cd ulmeval/dataset/utils/t2i_compbench/UniDet_eval/experts
mkdir -p expert_weights
cd expert_weights
wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt
pip install gdown
gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq
wget https://github.com/Karine-Huang/T2I-CompBench/raw/refs/heads/main/UniDet_eval/experts/obj_detection/datasets/label_spaces/learned_mAP+M.json
```

## Download bpe vocab
```bash
cd ulmeval/dataset/utils/t2i_compbench/CLIPScore_eval/clip
wget https://github.com/Karine-Huang/T2I-CompBench/raw/refs/heads/main/CLIPScore_eval/clip/bpe_simple_vocab_16e6.txt.gz
```

Put evaluation files in `LMUData/` and `export LMUData=/path/to/LMUData`
`. File names and structure should be something like:

```
LMUData
├── T2ICompBench_VAL.tsv
├── T2ICompBench_Color_VAL.tsv
├── T2ICompBench_Shape_VAL.tsv
├── T2ICompBench_Texture_VAL.tsv
├── T2ICompBench_Spatial_VAL.tsv
├── T2ICompBench_non_Spatial_VAL.tsv
└── T2ICompBench_Complex_VAL.tsv
```
