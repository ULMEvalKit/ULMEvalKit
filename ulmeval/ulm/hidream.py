import os.path as osp
import warnings

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from diffusers import HiDreamImagePipeline

from .base import BaseModel
from ..smp import splitlen


MODEL_REPOS = {
    "full": "HiDream-ai/HiDream-I1-Full",
    "dev": "HiDream-ai/HiDream-I1-Dev",
    "fast": "HiDream-ai/HiDream-I1-Fast",
}


class _HiDreamImageBase(BaseModel):
    """Shared implementation for HiDream-ai/HiDream-I1-* variants."""

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(
        self,
        model_key: str,  # "full" | "dev" | "fast"
        model_path: str | None = None,
        llama_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        *,
        default_guidance: float,
        default_steps: int,
        **kwargs,
    ):
        # ---------------- resolve HiDream model repo ----------------
        if model_path is None:
            assert model_key in MODEL_REPOS, f"Unsupported model_key: {model_key}"
            model_path = MODEL_REPOS[model_key]

        # Allow either local path or hub repo id
        assert osp.exists(model_path) or splitlen(model_path) == 2, \
            f"Invalid model_path: {model_path}"

        # Default generation config
        default_kwargs = dict(
            height=1024,
            width=1024,
            guidance_scale=default_guidance,
            num_inference_steps=default_steps,
            seed=None,             # if None, no fixed seed
            negative_prompt=None,  # optional, only passed if not None
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"{self.__class__.__name__} kwargs: {self.kwargs}")

        # ---------------- device / dtype ----------------
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"

        self.device = device
        self.model_key = model_key

        # ---------------- Llama tokenizer & text encoder ----------------
        # use_fast=False to avoid SentencePiece + tiktoken conversion issues
        self.tokenizer_4 = AutoTokenizer.from_pretrained(
            llama_id,
            use_fast=False,
        )
        self.text_encoder_4 = LlamaForCausalLM.from_pretrained(
            llama_id,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch_dtype,
        )

        # ---------------- HiDream pipeline ----------------
        self.pipe = HiDreamImagePipeline.from_pretrained(
            model_path,
            tokenizer_4=self.tokenizer_4,
            text_encoder_4=self.text_encoder_4,
            torch_dtype=torch_dtype,
        ).to(device)

    # ---------------- helpers ----------------
    def _prepare_prompt(self, message):
        # Same convention as QwenImage: collect text fields, ignore images
        prompt = "\n".join(m["value"] for m in message if m["type"] == "text")
        images = [m["value"] for m in message if m["type"] == "image"]
        if images:
            warnings.warn(
                f"{self.__class__.__name__} does not consume image inputs for generation; ignoring provided images."
            )
        return prompt

    def _build_generator(self):
        seed = self.kwargs.get("seed", None)
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def _call_pipe(self, prompt: str, num_images: int = 1):
        generator = self._build_generator()

        common_args = dict(
            height=self.kwargs.get("height", 1024),
            width=self.kwargs.get("width", 1024),
            guidance_scale=self.kwargs.get("guidance_scale"),
            num_inference_steps=self.kwargs.get("num_inference_steps"),
            generator=generator,
        )

        negative_prompt = self.kwargs.get("negative_prompt", None)
        if negative_prompt is not None:
            common_args["negative_prompt"] = negative_prompt

        if num_images > 1:
            common_args["num_images_per_prompt"] = num_images

        result = self.pipe(
            prompt,
            **common_args,
        )
        return result.images

    # ---------------- BaseModel API ----------------
    def generate_inner(self, message, dataset=None):
        prompt = self._prepare_prompt(message)
        images = self._call_pipe(prompt, num_images=1)
        return images[0]

    def batch_generate_inner(self, message, dataset, num_generations: int):
        prompt = self._prepare_prompt(message)
        images = self._call_pipe(prompt, num_images=num_generations)
        return images


class HiDreamImageFull(_HiDreamImageBase):
    """HiDream-I1 Full variant (5.0 guidance, 50 steps)."""

    def __init__(
        self,
        model_path: str | None = None,
        llama_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs,
    ):
        super().__init__(
            model_key="full",
            model_path=model_path,
            llama_id=llama_id,
            default_guidance=5.0,
            default_steps=50,
            **kwargs,
        )


class HiDreamImageDev(_HiDreamImageBase):
    """HiDream-I1 Dev (distilled) variant (0.0 guidance, 28 steps)."""

    def __init__(
        self,
        model_path: str | None = None,
        llama_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs,
    ):
        super().__init__(
            model_key="dev",
            model_path=model_path,
            llama_id=llama_id,
            default_guidance=0.0,
            default_steps=28,
            **kwargs,
        )


class HiDreamImageFast(_HiDreamImageBase):
    """HiDream-I1 Fast (distilled) variant (0.0 guidance, 16 steps)."""

    def __init__(
        self,
        model_path: str | None = None,
        llama_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs,
    ):
        super().__init__(
            model_key="fast",
            model_path=model_path,
            llama_id=llama_id,
            default_guidance=0.0,
            default_steps=16,
            **kwargs,
        )
