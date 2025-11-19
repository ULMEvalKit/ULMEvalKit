import os.path as osp
import warnings

import torch
from diffusers import DiffusionPipeline

from .base import BaseModel
from ..smp import splitlen


class QwenImage(BaseModel):
    """Text-to-image pipeline wrapper for Qwen/Qwen-Image."""

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path="Qwen/Qwen-Image", **kwargs):
        # Allow either local path or hub repo id
        assert osp.exists(model_path) or splitlen(model_path) == 2, \
            f"Invalid model_path: {model_path}"

        # Default generation config, mapped to the official example
        default_kwargs = dict(
            num_inference_steps=50,
            true_cfg_scale=4.0,
            negative_prompt=" ",
            # default aspect ratio: 16:9 from the official snippet
            width=1664,
            height=928,
            seed=None,  # optional, if None -> no fixed seed
            use_positive_magic=True,  # whether to append recommended suffix
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"QwenImage kwargs: {self.kwargs}")

        # Prepare device / dtype
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"

        # Official way: use DiffusionPipeline.from_pretrained
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        ).to(device)

        self.device = device

        # Recommended positive "magic" suffixes from the docs
        self.positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.",
            "zh": ", 超清，4K，电影级构图.",
        }

    def _prepare_prompt(self, message):
        prompt = "\n".join(m["value"] for m in message if m["type"] == "text")
        images = [m["value"] for m in message if m["type"] == "image"]
        if images:
            warnings.warn("QwenImage does not consume image inputs for generation; ignoring provided images.")
        return prompt

    @staticmethod
    def _looks_chinese(text: str) -> bool:
        # simple heuristic to choose zh/en suffix
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _build_generator(self):
        seed = self.kwargs.get("seed", None)
        if seed is None:
            return None
        # generator must live on same device as the model
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def _final_prompt(self, prompt: str) -> str:
        if not self.kwargs.get("use_positive_magic", True):
            return prompt
        lang = "zh" if self._looks_chinese(prompt) else "en"
        return prompt + self.positive_magic[lang]

    def generate_inner(self, message, dataset=None):
        prompt = self._prepare_prompt(message)
        prompt = self._final_prompt(prompt)
        generator = self._build_generator()

        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.kwargs.get("negative_prompt", " "),
            width=self.kwargs.get("width", 1664),
            height=self.kwargs.get("height", 928),
            num_inference_steps=self.kwargs.get("num_inference_steps", 50),
            true_cfg_scale=self.kwargs.get("true_cfg_scale", 4.0),
            generator=generator,
        )
        return result.images[0]

    def batch_generate_inner(self, message, dataset, num_generations):
        prompt = self._prepare_prompt(message)
        prompt = self._final_prompt(prompt)
        generator = self._build_generator()

        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.kwargs.get("negative_prompt", " "),
            width=self.kwargs.get("width", 1664),
            height=self.kwargs.get("height", 928),
            num_inference_steps=self.kwargs.get("num_inference_steps", 50),
            true_cfg_scale=self.kwargs.get("true_cfg_scale", 4.0),
            num_images_per_prompt=num_generations,
            generator=generator,
        )
        return result.images
