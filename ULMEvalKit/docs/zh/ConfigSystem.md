# 配置系统

默认情况下，ULMEvalKit 通过在 `run.py` 脚本中使用 `--model` 和 `--data` 参数设置模型名称（定义在 `/ulmeval/config.py` 中）和数据集名称（定义在 `ulmeval/dataset/__init__.py` 中）来启动评估。这种方法在大多数情况下简单高效，但当用户希望使用不同设置评估多个模型/数据集时，可能不够灵活。

为了解决这个问题，ULMEvalKit 提供了一个更灵活的配置系统。用户可以在一个 json 文件中指定模型和数据集的设置，然后通过 `run.py` 脚本的 `--config` 参数将配置文件路径传递给它。以下是一个示例配置 json：

```json
{
    "model": {
        "Janus-Pro-1B": {
            "class": "JanusPro",
            "model": "Janus-Pro-1B",
            "model_path": "<PATH>/deepseek-ai/Janus-Pro-1B",
            "temperature": 0.5
        },
        "Janus-Pro-7B": {}
    },
    "data": {
        "DPGBench": {
            "class": "DPGBench",
            "dataset": "DPGBench"
        },
        "T2ICompBench_non_Spatial_VAL": {
            "class": "T2ICompBench",
            "dataset": "T2ICompBench_non_Spatial_VAL"
        }
    }
}
```

对配置 json 的解释：

1.  目前我们支持两个字段：`model` 和 `data`，每个字段都是一个字典。字典的键是模型/数据集的名称（由用户设置），值是模型/数据集的设置。
2.  对于 `model` 中的项，其值是一个包含以下键的字典：
    -   `class`：模型的类名，应该是定义在 `ulmeval/ulm/__init__.py`（开源模型）或 `ulmeval/api/__init__.py`（API 模型）中的类名。
    -   其他 kwargs：其他 kwargs 是模型特定的参数，请参考模型类的定义以了解详细用法。例如，`model`、`temperature`、`model_path` 是 `JanusPro` 类的参数。值得注意的是，`model` 参数是大多数模型类所必需的。
    -   提示：在 `ulmeval/config.py` 的 `supported_ULM` 中定义的模型可以用作快捷方式，例如，`Janus-Pro-7B: {}` 等价于 `Janus-Pro-7B: {'class': 'JanusPro', 'model': 'Janus-Pro-7B', 'model_path': 'deepseek-ai/Janus-Pro-7B'}`。
3.  对于字典 `data`，我们建议用户使用官方数据集名称作为键（或键的一部分），因为我们经常根据数据集名称来确定后处理/评判设置。对于 `data` 中的项，其值是一个包含以下键的字典：
    -   `class`：数据集的类名，应该是定义在 `ulmeval/dataset/__init__.py` 中的类名。
    -   其他 kwargs：其他 kwargs 是数据集特定的参数，请参考数据集类的定义以了解详细用法。通常，`dataset` 参数是大多数数据集类所必需的。

将示例配置 json 保存为 `config.json`，您可以通过以下命令启动评估：

```bash
python run.py --config config.json
```

这将在工作目录 `$WORK_DIR` 下生成以下输出文件（遵循 `{$WORK_DIR}/{$MODEL_NAME}/{$MODEL_NAME}_{$DATASET_NAME}_*` 格式）：

- `$WORK_DIR/outputs/Janus-Pro-1B/T{date}_G{commit_id}/Janus-Pro-1B_DPGBench*`
- `$WORK_DIR/outputs/Janus-Pro-1B/T{date}_G{commit_id}/Janus-Pro-1B_T2ICompBench_non_Spatial_VAL*`
- `$WORK_DIR/outputs/Janus-Pro-7B/T{date}_G{commit_id}/Janus-Pro-1B_DPGBench*`
- `$WORK_DIR/outputs/Janus-Pro-7B/T{date}_G{commit_id}/Janus-Pro-1B_T2ICompBench_non_Spatial_VAL*`
...
