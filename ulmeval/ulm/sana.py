import os.path as osp
import warnings

import torch
from diffusers import SanaPipeline

from .base import BaseModel
from ..smp import splitlen


class _BaseSana15(BaseModel):
    """Minimal Sana-1.5 text-to-image wrapper.

    NOTE:
        Requires diffusers with SanaPipeline support, e.g.
        `pip install git+https://github.com/huggingface/diffusers`
    """

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path: str, **kwargs):
        # Allow either local path or hub repo id
        assert osp.exists(model_path) or splitlen(model_path) == 2, \
            f"Invalid model_path: {model_path}"

        # Default generation config (matches official examples)
        default_kwargs = dict(
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=20,
            seed=None,  # if None -> no fixed seed
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"Sana15 kwargs: {self.kwargs}")

        # Device / dtype
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"

        # SanaPipeline from diffusers
        self.pipe = SanaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        ).to(device)

        # Recommended component dtypes (from official snippet):
        #   - text_encoder + VAE in bf16 when possible
        if hasattr(self.pipe, "text_encoder"):
            self.pipe.text_encoder.to(torch_dtype)
        if hasattr(self.pipe, "vae"):
            self.pipe.vae.to(torch_dtype)

        self.device = device

    # ---- helpers ----

    def _prepare_prompt(self, message):
        """ULMEvalKit message -> plain text prompt."""
        prompt = "\n".join(m["value"] for m in message if m["type"] == "text")
        images = [m["value"] for m in message if m["type"] == "image"]
        if images:
            warnings.warn(
                "Sana15 pipeline is text-only; ignoring provided image inputs."
            )
        return prompt

    def _build_generator(self):
        seed = self.kwargs.get("seed", None)
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def _call_pipe(
        self,
        prompt: str,
        num_images_per_prompt: int = 1,
    ):
        generator = self._build_generator()
        result = self.pipe(
            prompt=prompt,
            height=self.kwargs.get("height", 1024),
            width=self.kwargs.get("width", 1024),
            guidance_scale=self.kwargs.get("guidance_scale", 4.5),
            num_inference_steps=self.kwargs.get("num_inference_steps", 20),
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )

        # diffusers pipelines usually return a PipelineOutput with `.images`
        if hasattr(result, "images"):
            return result.images
        # fallback to list-like output (as in some examples)
        return result[0]

    # ---- ULMEvalKit entry points ----

    def generate_inner(self, message, dataset=None):
        prompt = self._prepare_prompt(message)
        images = self._call_pipe(prompt, num_images_per_prompt=1)
        return images[0]

    def batch_generate_inner(self, message, dataset, num_generations):
        prompt = self._prepare_prompt(message)
        images = self._call_pipe(
            prompt,
            num_images_per_prompt=num_generations,
        )
        # `images` is already a list of PIL Images
        return images


class Sana15_1_6B(_BaseSana15):
    """SANA-1.5 1.6B @ 1024px wrapper."""

    def __init__(
        self,
        model_path: str = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        **kwargs,
    ):
        super().__init__(model_path=model_path, **kwargs)


class Sana15_4_8B(_BaseSana15):
    """SANA-1.5 4.8B @ 1024px wrapper."""

    def __init__(
        self,
        model_path: str = "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        **kwargs,
    ):
        super().__init__(model_path=model_path, **kwargs)
