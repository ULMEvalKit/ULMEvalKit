import os.path as osp
import warnings

import torch
from transformers import AutoModelForCausalLM

from .base import BaseModel
from ..smp import splitlen


class HunyuanImage(BaseModel):
    """Tencent HunyuanImage 3.0 text-to-image wrapper (Transformers-based)."""

    INSTALL_REQ = True
    INTERLEAVE = False
    allowed_types = ["text"]  # only text prompts are supported

    def __init__(self, model_path="tencent/HunyuanImage-3.0", **kwargs):
        # local dir or HF-style id (we still keep the old assert for compatibility)
        assert osp.exists(model_path) or splitlen(model_path) == 2, \
            f"Invalid model_path: {model_path}"

        # detach model loading-specific kwargs from generation kwargs
        attn_impl = kwargs.pop("attn_implementation", "sdpa")
        moe_impl = kwargs.pop("moe_impl", "eager")
        device_map = kwargs.pop("device_map", "auto")

        # default generation config (these go into generate_image)
        default_gen_kwargs = dict(
            diff_infer_steps=kwargs.pop("diff_infer_steps", 50),
            image_size=kwargs.pop("image_size", "auto"),
            seed=kwargs.pop("seed", None),
        )
        default_gen_kwargs.update(kwargs)
        self.kwargs = default_gen_kwargs
        warnings.warn(f"HunyuanImage kwargs: {self.kwargs}")

        # load HunyuanImage-3.0 model with transformers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map,
            moe_impl=moe_impl,
        )
        # tokenizer is required by the model's generate_image API
        self.model.load_tokenizer(model_path)

    def _prepare_prompt(self, message):
        # message is list[dict] from BaseModel.preproc_content
        prompt = "\n".join(m["value"] for m in message if m["type"] == "text")
        images = [m["value"] for m in message if m["type"] == "image"]
        if images:
            warnings.warn("HunyuanImage is text-to-image only; ignoring provided images.")
        return prompt

    def generate_inner(self, message, dataset=None):
        """Single-image generation to match BaseModel.generate / batch_generate_inner."""
        prompt = self._prepare_prompt(message)

        image = self.model.generate_image(
            prompt=prompt,
            stream=False,  # return final image directly
            diff_infer_steps=self.kwargs.get("diff_infer_steps", 50),
            image_size=self.kwargs.get("image_size", "auto"),
            seed=self.kwargs.get("seed", None),
        )
        # generate_image returns a PIL.Image in this usage
        return image
