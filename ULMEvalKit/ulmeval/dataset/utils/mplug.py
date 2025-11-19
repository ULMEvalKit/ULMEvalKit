import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from ...smp import *


class MPLUGModel(torch.nn.Module):
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en', device='gpu', **kwargs):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image, question):
        input_vqa = {'image': image, 'question': question}
        result = self.pipeline_vqa(input_vqa)
        return result['text']
