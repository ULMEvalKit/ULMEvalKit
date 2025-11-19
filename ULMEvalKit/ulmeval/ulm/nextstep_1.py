# ulmeval/ulm/nextstep_1.py
import os
import warnings
from typing import List, Dict, Any
from PIL import Image
import torch
from huggingface_hub import snapshot_download

from .base import BaseModel
from ..smp import splitlen


class NextStep1(BaseModel):
    INSTALL_REQ = True  # 用户需先安装 NextStep-1 代码库
    INTERLEAVE = False  # T2I 模型，不支持图像输入

    def __init__(self, **kwargs):
        try:
            from nextstep.models.pipeline_nextstep import NextStepPipeline
        except ImportError:
            raise ImportError(
                "NextStep-1 is not installed. Please run:\n"
                "git clone https://github.com/stepfun-ai/NextStep-1.git\n"
                "cd NextStep-1 && pip install -e ."
            )

        # 自动下载 VAE 并获取本地路径
        vae_hf_id = "stepfun-ai/NextStep-1-f8ch16-Tokenizer"
        print(f"[NextStep1Large] Downloading VAE from HuggingFace: {vae_hf_id}")
        vae_local_path = snapshot_download(
            repo_id=vae_hf_id,
            allow_patterns=["*.json", "*.pt", "*.py"]  # 只下载必要文件
        )
        print(f"[NextStep1Large] VAE path: {vae_local_path}")

        # 加载 pipeline
        self.pipeline = NextStepPipeline(
            model_name_or_path="stepfun-ai/NextStep-1-Large",
            vae_name_or_path=vae_local_path
        ).to("cuda", torch.bfloat16)

        # 默认生成参数
        default_cfg = dict(
            hw=(512, 512),
            num_sampling_steps=50,
            cfg=7.5,
            use_norm=True,
            seed=42
        )
        default_cfg.update(kwargs)
        self.gen_kwargs = default_cfg
        warnings.warn(f"[NextStep1Large] Using config: {self.gen_kwargs}")

    def _clean_prompt(self, prompt: str) -> str:
        """Remove leading <image> tokens for T2I tasks."""
        return prompt.replace("<image>", "").strip()

    def generate_inner(self, message: List[Dict[str, Any]], dataset=None) -> Image.Image:
        prompt = " ".join([msg["value"] for msg in message if msg["type"] == "text"])
        prompt = self._clean_prompt(prompt)
        if not prompt:
            raise ValueError("Empty prompt for NextStep-1-Large (T2I model).")

        output_imgs = self.pipeline.generate_image(
            captions=prompt,
            images=None,
            num_images_per_caption=1,
            positive_prompt=None,
            negative_prompt=(
                "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
                "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, "
                "watermark, username, blurry."
            ),
            hw=self.gen_kwargs["hw"],
            use_norm=self.gen_kwargs["use_norm"],
            cfg=self.gen_kwargs["cfg"],
            cfg_img=1.0,
            cfg_schedule="constant",
            timesteps_shift=1.0,
            num_sampling_steps=self.gen_kwargs["num_sampling_steps"],
            seed=self.gen_kwargs["seed"],
        )
        return output_imgs[0]

    def batch_generate_inner(self, message: List[Dict[str, Any]], dataset, num_generations: int) -> List[Image.Image]:
        prompt = " ".join([msg["value"] for msg in message if msg["type"] == "text"])
        prompt = self._clean_prompt(prompt)
        if not prompt:
            raise ValueError("Prompt is empty for NextStep-1-Large (T2I model).")

        output_imgs = self.pipeline.generate_image(
            captions=prompt,
            images=None,
            num_images_per_caption=num_generations,
            positive_prompt=None,
            negative_prompt=(
                "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
                "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, "
                "watermark, username, blurry."
            ),
            hw=self.gen_kwargs["hw"],
            use_norm=self.gen_kwargs["use_norm"],
            cfg=self.gen_kwargs["cfg"],
            cfg_img=1.0,
            cfg_schedule="constant",
            timesteps_shift=1.0,
            num_sampling_steps=self.gen_kwargs["num_sampling_steps"],
            seed=self.gen_kwargs["seed"],
        )
        return output_imgs
