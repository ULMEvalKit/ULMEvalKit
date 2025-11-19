import torch
import os
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen, listinstr
from PIL import Image
import logging
from einops import rearrange


class Flux(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='black-forest-labs/FLUX.1-dev', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        from diffusers import FluxPipeline

        default_kwargs = dict(
            guidance_scale=3.5,
            height=1024,
            width=1024,
            num_inference_steps=50,
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

        print('model_path: ', model_path)

        self.model = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_inner(self, message, dataset=None):
        prompt = [msg['value'] if msg['type'] == 'text' else "" for msg in message]
        prompt = '\n'.join(prompt)

        images = [msg['value'] for msg in message if msg['type'] == 'image']
        if len(images) > 0:
            warnings.warn(
                "FLUX does not support image inputs, ignoring them."
            )

        results = self.model(
            prompt=prompt,
            **self.kwargs
        )
        return results.images[0]

    def batch_generate_inner(self, message, dataset, num_generations):
        prompt = [msg['value'] if msg['type'] == 'text' else "" for msg in message]
        prompt = '\n'.join(prompt)

        images = [msg['value'] for msg in message if msg['type'] == 'image']
        if len(images) > 0:
            warnings.warn(
                "FLUX does not support image inputs, ignoring them."
            )

        results = self.model(
            prompt=prompt,
            num_images_per_prompt=num_generations,
            **self.kwargs
        )
        return results.images


class FluxKontext(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path='black-forest-labs/FLUX.1-Kontext-dev', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        from diffusers import FluxKontextPipeline

        default_kwargs = dict(
            guidance_scale=2.5,
            height=1024,
            width=1024,
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

        self.model = FluxKontextPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def generate_inner(self, message, dataset=None):
        from diffusers.utils import load_image

        prompt = [msg['value'] if msg['type'] == 'text' else "" for msg in message]
        prompt = '\n'.join(prompt)

        images = [msg['value'] for msg in message if msg['type'] == 'image']
        images = load_image(images[0])

        results = self.model(
            prompt=prompt,
            image=images,
            **self.kwargs
        )

        return results.images[0]

    def batch_generate_inner(self, message, dataset, num_generations):
        from diffusers.utils import load_image

        prompt = [msg['value'] if msg['type'] == 'text' else "" for msg in message]
        prompt = '\n'.join(prompt)

        images = [msg['value'] for msg in message if msg['type'] == 'image']
        images = load_image(images[0])

        results = self.model(
            prompt=prompt,
            image=images,
            num_images_per_prompt=num_generations,
            **self.kwargs
        )

        return results.images
