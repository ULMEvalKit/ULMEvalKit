import os
import sys
import os.path as osp
from pathlib import Path
import torch
from ...smp import *
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except:
    logger = get_logger('MMDetModel')
    logger.critical('MMDet is not installed. Please install it if you want to use detection models for evaluation.')


def get_gpu_num(model_name):
    model_name = model_name.lower()
    kws = {
        1: ['mask2former']
    }
    for k in [1]:
        for keyword in kws[k]:
            if keyword in model_name:
                return k
    return 1


validated_detectors = [
    'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco'
]
Auto_model = ['mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco']


class MMDetModel:

    def _get_context_length(self, model=None, model_path=None):
        self.logger.critical(
            'Detection models do not support context length. '
        )
        raise NotImplementedError

    def _get_context_length_robust(self, model=None, model_path=None):
        self.logger.critical(
            'Detection models do not support context length. '
        )
        raise NotImplementedError

    def __init__(self,
                 model_path,
                 config_path=None,
                 **kwargs):

        self.logger = get_logger('MMDetModel')

        self.explicit_device = kwargs.pop('device', None)
        if self.explicit_device is None:
            # If CUDA_VISIBLE_DEVICES is not properly set
            if ('CUDA_VISIBLE_DEVICES' not in os.environ
                    or os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'):
                num_gpu = get_gpu_num(model_path)
                gpu_offset = kwargs.pop('gpu_offset', 0)
                cuda_visible_devices = ','.join([str(i) for i in range(gpu_offset, gpu_offset + num_gpu)])
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

        self.model_path = model_path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(mmdet.__file__),
                "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
            )
        self.config_path = config_path
        device = self.explicit_device if self.explicit_device else 'auto'

        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        if ',' in cuda_devices:
            device_ids = [int(x) for x in cuda_devices.split(',')]
            _ = {i: i for i in range(len(device_ids))}
        else:
            _ = {'': 0}

        # Use the current working directory (assumed to be ULMEvalKit) for cache base
        cache_dir = Path.cwd() / "ulmeval" / "dataset" / "geneval"
        if not os.path.exists(os.path.join(cache_dir, f"{model_path}.pth")):
            raise FileNotFoundError(
                f"Model {model_path} not found in cache. Please download it "
                f"following https://github.com/djghosh13/geneval/blob/main/"
                f"evaluation/download_models.sh"
            )
        else:
            model_path = os.path.join(cache_dir, f"{model_path}.pth")

        model = init_detector(config_path, model_path)
        model = model.eval()

        if device != 'cpu':
            model = model.to(f'cuda:{device}' if isinstance(device, int) else 'cuda')

        torch.cuda.empty_cache()
        self.model = model
        for k, v in kwargs.items():
            self.logger.info(f'Following args will be used for generation (If not set specifically), {k}: {v}. ')
        self.kwargs = kwargs

    def generate(self, inputs, **kwargs):
        return inference_detector(self.model, inputs)
