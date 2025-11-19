# 快速开始

在运行评估脚本之前，您需要**配置** ULM 并正确设置模型路径。

之后，您可以使用单个脚本 `run.py` 同时推理和评估多个 ULM 和 Benchmark。

## 步骤 0. 安装 & 设置必要的密钥

**安装。**

```bash
git clone https://github.com/ULMEvalKit/ULMEvalKit.git
cd ULMEvalKit
pip install -e .
```

有关不同模型和 benchmark 的更多要求，请参见 [envs](./docs/envs/README.md)。

**设置密钥。**

为了使用 API 模型（GPT-4v, Gemini-Pro-V 等）进行推理，您需要先设置 API 密钥。
- 您可以将所需的密钥放在 `$ULMEvalKit/.env` 文件中，或者直接将它们设置为环境变量。如果您选择创建 `.env` 文件，其内容将如下所示：

  ```bash
  # .env 文件，将其放在 $ULMEvalKit 目录下
  # 使用 Google Cloud 后端的 Gemini
  GOOGLE_API_KEY=
  # OpenAI API
  OPENAI_API_KEY=
  OPENAI_API_BASE=
  # 您也可以在评估阶段为调用 API 模型设置代理
  EVAL_PROXY=
  ```

- 在空白处填入您的 API 密钥（如果需要）。这些 API 密钥将在进行推理和评估时自动加载。

## 步骤 1. 配置

**ULM 配置**：所有 ULM 都在 `ulmeval/config.py` 中配置。在评估期间，您应使用 `ulmeval/config.py` 中 `supported_ULM` 指定的模型名称来选择 ULM。

## 步骤 2. 评估

我们使用 `run.py` 进行推理和评估，并且可以使用 `--mode infer` 仅执行推理。同时，提供了 `evaluate.py` 来仅评估由 `run.py` 生成的结果。

要使用该脚本，您可以使用 `$ULMEvalKit/run.py`：

**参数**

- `--data (list[str])`：设置在 ULMEvalKit 中支持的数据集名称（名称可以在代码库的 README 中找到）。
- `--model (list[str])`：设置在 ULMEvalKit 中支持的 ULM 名称（在 `ulmeval/config.py` 的 `supported_ULM` 中定义）。
- `--mode (str, 默认为 'all', 可选值为 ['all', 'infer'])`：
  - 当 `mode` 设置为 `all` 时，将同时执行推理和评估；
  - 当设置为 `infer` 时，将仅执行推理。这在 ULM 和 benchmark 的环境不兼容时非常有用。
- `--api-nproc (int, 默认为 4)`：OpenAI API 调用的线程数。
- `--work-dir (str, 默认为 '.')`：保存评估结果的目录。
- `--num-generations (int, 默认为 None)`：每个提示的生成次数。默认为 None，表示生成次数由数据集决定。如果数据集未指定生成次数，则默认为 1。

**评估命令**

您可以使用 `python` 或 `torchrun` 运行脚本：

```bash
# 当使用 `python` 运行时，仅实例化一个 ULM 实例。

# OmniGen2 在 T2ICompBench_non_Spatial_VAL 上，使用 T2ICompBench 的默认生成次数 (10)，推理和评估
python run.py --data T2ICompBench_non_Spatial_VAL --model OmniGen2

# Janus-1.3B 在 DPGBench 上，每个提示生成 5 次，仅推理
python run.py --data DPGBench --model Janus-1.3B --num-generations 5 --mode infer

# Janus-1.3B 在 DPGBench 上，仅评估
python evaluate.py --data DPGBench --model Janus-1.3B --result-file ./outputs/Janus-1.3B/T{date}_G{commit_id}/Janus-1.3B_DPGBench.pkl

# 当使用 `torchrun` 运行时，每个 GPU 上实例化一个 ULM 实例。这可以加速推理。
# 但是，这仅适用于消耗少量 GPU 内存的 ULMs。

# Janus-Pro-7B 在 T2ICompBench_non_Spatial_VAL 上，每个提示生成 10 次。在具有 8 个 GPU 的节点上。
torchrun --nproc_per_node=8 run.py --data T2ICompBench_non_Spatial_VAL --model Janus-Pro-7B --num-generations 10
```

评估结果将作为日志打印出来。此外，**结果文件**也将在目录 `$YOUR_WORKING_DIRECTORY/{model_name}` 中生成。以 `.csv` 或 `.pkl` 结尾的文件包含评估的指标。
