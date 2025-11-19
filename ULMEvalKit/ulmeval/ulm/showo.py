import sys
import os
from PIL import ImageDraw
import torch
import numpy as np
from PIL import Image
from ..smp import *
from .base import BaseModel
from transformers import set_seed


class Showo(BaseModel):

    def __init__(self, mode, model_path, **kwargs):

        if model_path is None:
            raise ValueError("model_path is required. Please provide the path to Show-o project.")
        self.model_path = model_path

        if mode == 'showo_demo_512x512':
            cfg = 'showo_demo_512x512.yaml'
        elif mode == 'showo_demo_w_clip_vit_512x512':
            cfg = 'showo_demo_w_clip_vit_512x512.yaml'
        elif mode == 'showo_demo_w_clip_vit':
            cfg = 'showo_demo_w_clip_vit.yaml'
        elif mode == 'showo_demo':
            cfg = 'showo_demo.yaml'
        else:
            raise NotImplementedError

        self.mode = mode
        self.cfg = osp.join(
            self.model_path, 'configs', cfg
        )
        self.kwargs = kwargs
        sys.path.insert(0, model_path)
        from models import Showo as ShowoModel, MAGVITv2
        from models.sampling import get_mask_chedule
        from transformers import AutoTokenizer
        from training.utils import get_config
        from training.prompting_utils import UniversalPrompting
        from training.prompting_utils import create_attention_mask_predict_next

        self.create_attention_mask_predict_next = create_attention_mask_predict_next
        self.get_mask_chedule = get_mask_chedule
        cfg_path = os.path.join(self.model_path, 'configs', cfg)
        sys.argv = ['script', f'config={cfg_path}']
        config = get_config()
        self.config = config
        tokenizer_model_path = config.model.showo.llm_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_path,
            padding_side="left"
        )

        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                            "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob
        )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pretrained_path = config.model.showo.pretrained_model_path
        self.model = ShowoModel.from_pretrained(
            pretrained_path,
            device_map=None,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(self.device)
        vq_model_type = config.model.vq_model.type
        if vq_model_type == "magvitv2":
            self.vq_model = MAGVITv2.from_pretrained(
                config.model.vq_model.vq_model_name,
                device_map=None,
                low_cpu_mem_usage=False,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"vq_model_type {vq_model_type} not supported.")
        self.vq_model.eval()
        self.vq_model.to(self.device)
        self.kwargs = {}
        guidance_scale = 1.75
        self.guidance_scale = guidance_scale

    def batch_generate_inner(self, message, dataset=None, num_generations=1):
        if isinstance(message, list):
            prompt = ""
            for item in message:
                if isinstance(item, dict) and item.get('type') == 'text':
                    prompt += item.get('value', '')
                elif isinstance(item, str):
                    prompt += item
        else:
            prompt = str(message)

        prompts = [prompt] * num_generations

        with torch.no_grad():
            image_tokens = torch.ones(
                (len(prompts), self.config.model.showo.num_vq_tokens),
                dtype=torch.long, device=self.device
            ) * self.model.config.mask_token_id

            input_ids, _ = self.uni_prompting((prompts, image_tokens), 't2i_gen')
            input_ids = input_ids.to(self.device)

            if self.guidance_scale > 0:
                uncond_input_ids, _ = self.uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
                attention_mask = self.create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True,
                )
                # Ensure attention_mask is on the correct device and dtype for SDPA
                attention_mask = attention_mask.to(self.device)
                if attention_mask.dtype.is_floating_point:
                    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    attention_mask = attention_mask.to(dtype=target_dtype)
            else:
                attention_mask = self.create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True,
                )
                # Ensure attention_mask is on the correct device and dtype for SDPA
                attention_mask = attention_mask.to(self.device)
                if attention_mask.dtype.is_floating_point:
                    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    attention_mask = attention_mask.to(dtype=target_dtype)
                uncond_input_ids = None

            if self.config.get("mask_schedule", None) is not None:
                schedule = self.config.mask_schedule.schedule
                args = self.config.mask_schedule.get("params", {})
                mask_schedule = self.get_mask_chedule(schedule, **args)
            else:
                mask_schedule = self.get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = self.model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=self.guidance_scale,
                    temperature=1.0,
                    timesteps=16,
                    noise_schedule=mask_schedule,
                    noise_type="mask",
                    seq_len=self.config.model.showo.num_vq_tokens,
                    uni_prompting=self.uni_prompting,
                    config=self.config,
                )

            gen_token_ids = torch.clamp(
                gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0
            )
            images = self.vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            generated_images = [Image.fromarray(image) for image in images]
            return generated_images

    def generate_inner(self, message, dataset=None):
        # Extract prompt from message
        if isinstance(message, list):
            prompt = ""
            for item in message:
                if isinstance(item, dict) and item.get('type') == 'text':
                    prompt += item.get('value', '')
                elif isinstance(item, str):
                    prompt += item
        else:
            prompt = str(message)

        prompts = [prompt] * 1

        with torch.no_grad():
            image_tokens = torch.ones(
                (len(prompts), self.config.model.showo.num_vq_tokens),
                dtype=torch.long, device=self.device
            ) * self.model.config.mask_token_id

            input_ids, _ = self.uni_prompting(([prompt], image_tokens), 't2i_gen')
            input_ids = input_ids.to(self.device)

            if self.guidance_scale > 0:
                uncond_input_ids, _ = self.uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
                attention_mask = self.create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True,
                )
                # Ensure attention_mask is on the correct device and dtype for SDPA
                attention_mask = attention_mask.to(self.device)
                if attention_mask.dtype.is_floating_point:
                    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    attention_mask = attention_mask.to(dtype=target_dtype)
            else:
                attention_mask = self.create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True,
                )
                # Ensure attention_mask is on the correct device and dtype for SDPA
                attention_mask = attention_mask.to(self.device)
                if attention_mask.dtype.is_floating_point:
                    target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    attention_mask = attention_mask.to(dtype=target_dtype)
                uncond_input_ids = None

            if self.config.get("mask_schedule", None) is not None:
                schedule = self.config.mask_schedule.schedule
                args = self.config.mask_schedule.get("params", {})
                mask_schedule = self.get_mask_chedule(schedule, **args)
            else:
                mask_schedule = self.get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = self.model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=self.guidance_scale,
                    temperature=1.0,
                    timesteps=16,
                    noise_schedule=mask_schedule,
                    noise_type="mask",
                    seq_len=self.config.model.showo.num_vq_tokens,
                    uni_prompting=self.uni_prompting,
                    config=self.config,
                )

            gen_token_ids = torch.clamp(
                gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0
            )
            images = self.vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            generated_images = [Image.fromarray(image) for image in images]
            return generated_images[0] if generated_images else None
