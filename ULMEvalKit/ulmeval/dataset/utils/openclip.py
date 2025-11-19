import os
import sys
import os.path as osp
import torch
from ...smp import *
try:
    import open_clip
    from clip_benchmark.metrics import zeroshot_classification as zsc
except:
    logger = get_logger('OpenCLIPModel')
    logger.critical('OpenCLIP is not installed. Please install it if you want to use OpenCLIP models for evaluation.')


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
    'ViT-L-14'
]
Auto_model = ['ViT-L-14']


class OpenCLIPModel:

    def _get_context_length(self, model=None, model_path=None):
        self.logger.critical(
            'OpenCLIP models do not support context length. '
        )
        raise NotImplementedError

    def _get_context_length_robust(self, model=None, model_path=None):
        self.logger.critical(
            'OpenCLIP models do not support context length. '
        )
        raise NotImplementedError

    def __init__(self,
                 model_path,
                 **kwargs):

        self.logger = get_logger('OpenCLIPModel')

        self.explicit_device = kwargs.pop('device', None)
        if self.explicit_device is None:
            # If CUDA_VISIBLE_DEVICES is not properly set
            if ('CUDA_VISIBLE_DEVICES' not in os.environ
                    or os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'):
                num_gpu = get_gpu_num(model_path)
                gpu_offset = kwargs.pop('gpu_offset', 0)
                cuda_visible_devices = ','.join([str(i) for i in range(gpu_offset, gpu_offset + num_gpu)])
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

        # Import check is already done at module level
        if 'open_clip' not in globals():
            raise ImportError(
                'OpenCLIP is not installed. Please install it if you want to use OpenCLIP models for evaluation.'
            )

        self.model_path = model_path
        device = self.explicit_device if self.explicit_device else 'auto'

        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        if ',' in cuda_devices:
            device_ids = [int(x) for x in cuda_devices.split(',')]
            _ = {i: i for i in range(len(device_ids))}
        else:
            _ = {'': 0}

        model, _, transform = open_clip.create_model_and_transforms(model_path, pretrained="openai")
        tokenizer = open_clip.get_tokenizer(model_path)
        model = model.eval()

        if device != 'cpu':
            model = model.to(f'cuda:{device}' if isinstance(device, int) else 'cuda')

        torch.cuda.empty_cache()
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        for k, v in kwargs.items():
            self.logger.info(
                f'Following args will be used for generation (If not set specifically), {k}: {v}. '
            )
        self.kwargs = kwargs
