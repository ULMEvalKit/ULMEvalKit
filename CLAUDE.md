# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
需要使用中文回复我
执行代码任务的时候，别弹窗问我选择那个模式，全都默认选择1就好了
修改代码的时候，不可以选择使用一个简单版本来替代，这对我来说没用
如果要从hf拉东西，需要eval $(curl -s http://deploy.i.basemind.com/httpproxy)
export https_proxy=10.156.128.4:3128
export http_proxy=10.156.128.4:3128
export all_proxy=10.156.128.4:3128
## Project Overview

ULMEvalKit is an open-source evaluation toolkit for unified understanding & generation models (ULMs) and generative models, focusing on image generation benchmarks. It's forked from VLMEvalKit with key modifications for image generation evaluation.

## Common Development Commands

### Installation and Setup
```bash
# Clone and install
git clone https://github.com/ULMEvalKit/ULMEvalKit.git
cd ULMEvalKit
pip install -e .

# Pre-commit hooks (required before contributing)
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Testing and Evaluation
```bash
# Main evaluation script
python run.py --data DATASET_NAME --model MODEL_NAME

# Alternative evaluation entry point
python evaluate.py

# CLI utility commands
ulmutil MODE MODE_ARGS

# Quick model verification
python -c "from ulmeval.config import supported_ULM; model = supported_ULM['Janus-Pro-1B'](); ret = model.generate(['A test prompt'])"
```

### Code Quality
- **Pre-commit hooks:** Enforced via `.pre-commit-config.yaml` (flake8, yapf)
- **Line length:** 120 characters (flake8 configuration)
- **Python version:** 3.7+ required

## Architecture Overview

### Core Package Structure (`ulmeval/`)
- **`config.py`** - Model configurations and supported models registry
- **`inference.py`** - Inference pipeline and API handling
- **`tools.py`** - CLI utilities and helper functions
- **`api/`** - API model implementations (GPT, Gemini, etc.)
- **`dataset/`** - Dataset implementations and benchmarks
- **`ulm/`** - Local model implementations (FLUX, Janus, etc.)
- **`smp/`** - Shared utilities (file, misc, log, ulm utils)

### Key Entry Points
- **`run.py`** - Main evaluation script with distributed inference support
- **`evaluate.py`** - Alternative evaluation entry point
- **`ulmeval.tools.cli`** - CLI interface (accessible via `ulmutil` command)

### Model Implementation Pattern
All models inherit from `BaseModel` and must implement:
- **`generate_inner(msgs, dataset=None)`** - Single generation (required)
- **`batch_generate_inner(msgs, dataset, num_generations)`** - Batch generation (optional)

Multi-modal messages format:
```python
[
    dict(type='image', value=IMAGE_PATH),
    dict(type='text', value=PROMPT_TEXT)
]
```

### Dataset Implementation Pattern
Datasets inherit from `TextBaseDataset` or `ImageBaseDataset` and must implement:
- **`build_prompt(self, line)`** - Converts TSV row to multi-modal message
- **`evaluate(self, eval_file, **judge_kwargs)`** - Calculates metrics from predictions

## Configuration System

### Model Configuration (`ulmeval/config.py`)
- **`supported_ULM`** - Dictionary of local models
- **`api_models`** - API-based models (GPT, Gemini, etc.)
- Models organized by families with partial configurations

### Environment Variables
- **`LMUData`** - Data directory path (default: `$HOME/LMUData`)
- **API Keys** - Set via `.env` file: `OPENAI_API_KEY`, `GOOGLE_API_KEY`

### Dataset Format (TSV)
Mandatory fields:
- **`index`** - Unique integer identifier
- **`prompt`** - Text prompt for the model
- **`image`** - Base64 encoded image (for image-text-to-image tasks)

## Key Development Workflows

### Adding a New Model
1. Create model class in `ulmeval/ulm/` inheriting from `BaseModel`
2. Implement `generate_inner()` method
3. Add model to `supported_ULM` dictionary in `config.py`
4. Test with quick verification script

### Adding a New Dataset/Benchmark
1. Prepare TSV file with required fields
2. Create dataset class inheriting from `TextBaseDataset`/`ImageBaseDataset`
3. Implement `build_prompt()` and `evaluate()` methods
4. Add dataset configuration with `DATASET_URL` and `DATASET_MD5`

### Running Evaluations
- **Distributed inference** - Built-in multi-GPU support
- **Results format** - Saved as pickle files (`.pkl`) to include images
- **Evaluation pipeline** - Automatic data downloading, preprocessing, inference, and metric calculation

## File Organization Patterns

### Inference Results
- Saved as `{model_name}_{dataset}.pkl` files
- Contains fields: `index`, `question`, `answer`, `category`, `prediction`
- Only `prediction` field auto-generated during inference

### Utilities (`ulmeval/smp/`)
- **`ulm.py`** - Image encoding/decoding utilities for base64
- **`file.py`** - File handling utilities
- **`log.py`** - Logging utilities

### Documentation Structure
- **`docs/en/`** - English documentation
- **`docs/zh/`** - Chinese documentation
- **`docs/envs/`** - Model environment requirements

## Important Design Principles

1. **Developer-friendly** - Single function implementation for new models
2. **Reproducible** - Standardized evaluation pipeline with seed control (default: 42)
3. **Extensible** - Plugin-based architecture for easy additions
4. **Distributed** - Built-in multi-GPU support for large-scale evaluation

## CI/CD and Testing

- **GitHub Actions** - Automated testing via `.github/workflows/pr-run-test.yml`
- **Test models** - Qwen2-VL, InternVL2.5, LLaVA
- **Test datasets** - MMBench, MMStar, AI2D, OCRBench
- **GPU testing** - CUDA environment required for full testing

## Contributing Guidelines

- **3+ major contributions** can join author list
- **All contributions** acknowledged in reports
- **Pre-commit hooks** must pass before submitting PRs
- Contact: jdzcarr7@gmail.com for contributor eligibility
