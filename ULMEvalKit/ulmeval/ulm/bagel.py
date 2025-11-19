import os
from copy import deepcopy
import requests
from io import BytesIO

import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen, listinstr
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from safetensors.torch import load_file
import logging


class Bagel(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path='ByteDance-Seed/BAGEL-7B-MoT', **kwargs):

        assert osp.exists(model_path) or splitlen(model_path) == 2

        try:
            from data.transforms import ImageTransform
            from data.data_utils import pil_img2rgb, add_special_tokens
            from modeling.bagel import (
                BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
            )
            from modeling.qwen2 import Qwen2Tokenizer
            from modeling.bagel.qwen2_navit import NaiveCache
            from modeling.autoencoder import load_ae

            from inferencer import InterleaveInferencer

        except Exception as err:
            logging.critical(
                "Please install Bagel from https://github.com//ByteDance-Seed/Bagel"
            )
            raise err

        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        self.vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            self.language_model = Qwen2ForCausalLM(llm_config)
            self.vit_model = SiglipVisionModel(vit_config)
            self.model = Bagel(self.language_model, self.vit_model, config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)

        max_mem_per_gpu = "40GiB"  # Modify it according to your GPU setting.
        # On an A100, 80 GiB is sufficient to load on a single GPU.

        self.device_map = infer_auto_device_map(
            self.model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = self.device_map.get(same_device_modules[0], "cuda0")
            for k in same_device_modules:
                if k in self.device_map:
                    self.device_map[k] = first_device
                else:
                    self.device_map[k] = "cuda:0"
        else:
            first_device = self.device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in self.device_map:
                    self.device_map[k] = first_device

        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=self.device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )

        self.model = self.model.eval()

        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )

        kwargs_default = {
            "max_think_token_n": 1000,
            "do_sample": False,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global"
        }

        self.kwargs = kwargs_default
        kwargs_default.update(kwargs)

        self.file_root = osp.dirname(__file__)
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

    def generate_inner(self, message, dataset=None):
        prompt = [msg['value'] if msg['type'] == 'text' else "" for msg in message]
        prompt = '\n'.join(prompt)

        images = [msg['value'] for msg in message if msg['type'] == 'image']
        output_dict = {}

        if len(images) == 0:
            output_dict = self.inferencer(text=prompt, **self.kwargs)

        else:
            if len(images) > 1:
                warnings.warn(
                    'Bagel only support single image as input, take first image as input'
                )
            image = Image.open(images[0])
            output_dict = self.inferencer(image=image, text=prompt, **self.kwargs)

        return output_dict['image']
