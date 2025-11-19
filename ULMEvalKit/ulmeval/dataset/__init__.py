import warnings

from .utils import *
from ..smp import *
from .t2i_compbench import T2ICompBench
from .geneval import GenEval
from .dpgbench import DPGBench
from .wise import WISE
from .imagine_bench import ImagineBench
from .geneval_pp import GenEvalPP
from .omni_context import OmniContext
from .genai_bench import GenAI_Bench

IMAGE_DATASET = [
    OmniContext
]

TEXT_DATASET = [
    T2ICompBench, DPGBench, WISE, ImagineBench, GenEvalPP, GenAI_Bench, GenEval
]

DATASET_CLASSES = IMAGE_DATASET + TEXT_DATASET
SUPPORTED_DATASETS = []
for DATASET_CLS in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(DATASET_CLS.supported_datasets())


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'TYPE'):
                return cls.TYPE
    # Have to add specific routine to handle ConcatDataset
    if 'openended' in dataset.lower():
        return 'VQA'
    warnings.warn(f'Dataset {dataset} is a custom one and not annotated as `openended`, will treat as {default}. ')  # noqa: E501
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    if dataset is None:
        warnings.warn(f'Dataset is not specified, will treat modality as {default}. ')
        return default
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'MODALITY'):
                return cls.MODALITY
    warnings.warn(f'Dataset {dataset} is a custom one, will treat modality as {default}. ')
    return default


def build_dataset(dataset_name, **kwargs):
    for cls in DATASET_CLASSES:
        if dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')
    return None


def infer_dataset_basename(dataset_name):
    basename = "_".join(dataset_name.split("_")[:-1])
    return basename


__all__ = [
    'build_dataset', 'build_judge', 'DEBUG_MESSAGE'
] + [cls.__name__ for cls in DATASET_CLASSES]
