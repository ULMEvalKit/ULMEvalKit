import os
from ...smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)


def build_judge(**kwargs):
    from ...api import OpenAIWrapper
    from .mmdet_model import MMDetModel
    from .openclip import OpenCLIPModel
    from .mplug import MPLUGModel

    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()
    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        model_map = {
            'gpt-4o': 'gpt-4o-2024-05-13',
            'mask2former': 'mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco',
            'openclip-vit-l-14': 'ViT-L-14',
            'mplug': 'damo/mplug_visual-question-answering_coco_large_en'
        }
        model_version = model_map[model] if model in model_map else model
    else:
        model_version = LOCAL_LLM

    if model == 'mask2former':
        model = MMDetModel(model_version, **kwargs)
    elif model == 'openclip-vit-l-14':
        model = OpenCLIPModel(model_version, **kwargs)
    elif model == 'mplug':
        model = MPLUGModel(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You cam see the specific error if the API call fails.
"""
