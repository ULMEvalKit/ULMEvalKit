# Config System

By default, ULMEvalKit launches the evaluation by setting the model name(s) (defined in `/ulmeval/config.py`) and dataset name(s) (defined in `ulmeval/dataset/__init__.py`) in the `run.py` script with the `--model` and `--data` arguments. Such approach is simple and efficient in most scenarios, however, it may not be flexible enough when the user wants to evaluate multiple models / datasets with different settings.

To address this, ULMEvalKit provides a more flexible config system. The user can specify the model and dataset settings in a json file, and pass the path to the config file to the `run.py` script with the `--config` argument. Here is a sample config json:

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

Explanation of the config json:

1. Now we support two fields: `model` and `data`, each of which is a dictionary. The key of the dictionary is the name of the model / dataset (set by the user), and the value is the setting of the model / dataset.
2. For items in `model`, the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class name defined in `ulmeval/ulm/__init__.py` (open-source models) or `ulmeval/api/__init__.py` (API models).
    - Other kwargs: Other kwargs are model-specific parameters, please refer to the definition of the model class for detailed usage. For example, `model`, `temperature`, `model_path` are arguments of the `JanusPro` class. It's noteworthy that the `model` argument is required by most model classes.
    - Tip: The defined model in the `supported_ULM` of `ulmeval/config.py` can be used as a shortcut, for example, `Janus-Pro-7B: {}` is equivalent to `Janus-Pro-7B: {'class': 'JanusPro', 'model': 'Janus-Pro-7B', 'model_path': 'deepseek-ai/Janus-Pro-7B'}`
3. For the dictionary `data`, we suggest users to use the official dataset name as the key (or part of the key), since we frequently determine the post-processing / judging settings based on the dataset name. For items in `data`, the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class name defined in `ulmeval/dataset/__init__.py`.
    - Other kwargs: Other kwargs are dataset-specific parameters, please refer to the definition of the dataset class for detailed usage. Typically, the `dataset` argument is required by most dataset classes.
Saving the example config json to `config.json`, you can launch the evaluation by:

```bash
python run.py --config config.json
```

That will generate the following output files under the working directory `$WORK_DIR` (Following the format `{$WORK_DIR}/{$MODEL_NAME}/{$MODEL_NAME}_{$DATASET_NAME}_*`):

- `$WORK_DIR/outputs/Janus-Pro-1B/T{date}_G{commit_id}/Janus-Pro-1B_DPGBench*`
- `$WORK_DIR/outputs/Janus-Pro-1B/T{date}_G{commit_id}/Janus-Pro-1B_T2ICompBench_non_Spatial_VAL*`
- `$WORK_DIR/outputs/Janus-Pro-7B/T{date}_G{commit_id}/Janus-Pro-1B_DPGBench*`
- `$WORK_DIR/outputs/Janus-Pro-7B/T{date}_G{commit_id}/Janus-Pro-1B_T2ICompBench_non_Spatial_VAL*`
...
