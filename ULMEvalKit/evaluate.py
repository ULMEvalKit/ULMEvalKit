import json
import os
import subprocess
from functools import partial


# GET the number of GPUs on the node without importing libs like torch
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        return []


RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))

GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert NGPU >= LOCAL_WORLD_SIZE, "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [str(i) for i in GPU_LIST[DEVICE_START_IDX: DEVICE_START_IDX + GPU_PER_PROC]]
    CUDA_VISIBLE_DEVICES = ','.join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    print(
        f'RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE},'
        f'LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}'
    )


from ulmeval.config import supported_ULM
from ulmeval.dataset import build_dataset
from ulmeval.smp import *


# Make WORLD_SIZE invisible when build models
def build_model_from_config(cfg, model_name, use_vllm=False):
    import ulmeval.api
    import ulmeval.ulm
    ws_bak = os.environ.pop('WORLD_SIZE', None)

    config = cp.deepcopy(cfg[model_name])
    if use_vllm:
        config['use_vllm'] = use_vllm
    if 'class' not in config:
        return supported_ULM[model_name](**config)
    cls_name = config.pop('class')
    if hasattr(ulmeval.api, cls_name):
        model = getattr(ulmeval.api, cls_name)(**config)
    elif hasattr(ulmeval.ulm, cls_name):
        model = getattr(ulmeval.ulm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `ulmeval.api` or `ulmeval.ulm`')

    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak
    return model


def build_dataset_from_config(cfg, dataset_name):
    import ulmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(ulmeval.dataset, cls_name):
        cls = getattr(ulmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `ulmeval.dataset`')


def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `ulmeval/config.py` file.
    To find all supported dataset names, please refer to the `ulmeval/dataset/__init__.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
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
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `ulmeval.ulm` or `ulmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_ULM` of `ulmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `ulmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API ulms, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--result-file', type=str, nargs='+', help='Names of Result Files')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # API Kwargs, Apply to API ulms and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API ulms')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed ulm inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    parser.add_argument(
        '--use-vllm', action='store_true', help='use vllm to generate, the flag is only supported in Llama4 for now')
    parser.add_argument('--num-generations', type=int, default=1, help='number of generations for each prompt')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')
    args = parse_args()
    use_config, cfg = False, None
    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'

    if not use_config:
        for k, v in supported_ULM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_ULM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_ULM[k] = v

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    for model_name, result_file in zip(args.model, args.result_file):

        for _, dataset_name in enumerate(args.data):
            if WORLD_SIZE > 1:
                dist.barrier()

            try:
                if use_config:
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                else:
                    dataset_kwargs = {}

                    # If distributed, first build the dataset on the main process for doing preparation works
                    if WORLD_SIZE > 1:
                        if RANK == 0:
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                        dist.barrier()

                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Set the judge kwargs first before evaluation or dumping

                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge

                elif listinstr(['WISE','GenEvalPP','ImagineBench','OmniContext'], dataset_name):
                    judge_kwargs['model'] = 'gpt-4o'

                if RANK == 0:
                    logger.info(judge_kwargs)

                if WORLD_SIZE > 1:
                    dist.barrier()

                # Only RANK 0 handles the evaluation part
                if dataset_name in [
                    'GenEval',
                    'T2ICompBench_VAL',
                    'T2ICompBench_non_Spatial_VAL',
                    'T2ICompBench_Spatial_VAL',
                    'T2ICompBench_Color_VAL',
                    'T2ICompBench_Shape_VAL',
                    'T2ICompBench_Texture_VAL',
                    'T2ICompBench_Complex_VAL',
                    'DPGBench',
                    'WISE_all',
                    'GenEvalPP',
                    'ImagineBench',
                    'OmniContext'
                ]:  # use all rank to eval

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    if WORLD_SIZE > 1:
                        dist.barrier()
                    # Perform the Evaluation

                    eval_results = dataset.evaluate(result_file, **judge_kwargs)

                    if RANK == 0:
                        # Display Evaluation Results in Terminal
                        if eval_results is not None:
                            assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                            logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                            logger.info('Evaluation Results:')
                            if isinstance(eval_results, dict):
                                logger.info('\n' + json.dumps(eval_results, indent=4))
                            elif isinstance(eval_results, pd.DataFrame):
                                if len(eval_results) < len(eval_results.columns):
                                    eval_results = eval_results.T
                                logger.info('\n' + tabulate(eval_results))

                        # Restore the proxy
                        if eval_proxy is not None:
                            proxy_set(old_proxy)

                elif RANK == 0:
                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        logger.info('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            logger.info('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info('\n' + tabulate(eval_results))

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

            except Exception as e:
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                 'skipping this combination.')
                continue

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    load_env()
    main()
