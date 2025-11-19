import os
from copy import deepcopy

import torch
import os.path as osp
import sys
import warnings
from .base import BaseModel
from ..smp import splitlen, listinstr
from ..dataset import DATASET_TYPE
import PIL.Image

import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

import torchvision.transforms as T

import logging


class T2IR1(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='CaraJ/T2I-R1', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        try:
            t2ir1_root = os.environ.get('T2IR1_ROOT', '.')
            sys.path.insert(0, t2ir1_root)
            from janus.models import MultiModalityCausalLM, VLChatProcessor

        except Exception as err:
            logging.critical(
                'Please first clone T2IR1 from source codes in: https://github.com/CaraJ7/T2I-R1 and set up the environment variable T2IR1_ROOT as "T2I-R1/src/t2i-r1/src"')  # noqa: E501
            raise err

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).cuda().eval()

        self.cot_prompt = '''You are asked to generate an image based on this prompt: "{}"
Provide a brief, precise visualization of all elements in the prompt. Your description should:
1. Include every object mentioned in the prompt
2. Specify visual attributes (color, number, shape, texture) if specified in the prompt
3. Clarify relationships (e.g., spatial) between objects if specified in the prompt
4. Be concise (50 words or less)
5. Focus only on what's explicitly stated in the prompt
6. Do not elaborate beyond the attributes or relationships specified in the prompt
Do not miss objects. Output your visualization directly without explanation: '''  # noqa: E501

        self.system_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.'  # noqa: E501

        default_kwargs = dict(
            temperature=1,
            cfg_weight=5,
            parallel_size=1,
            image_token_num_per_image=576,
            img_size=384,
            patch_size=16
        )

        default_kwargs.update(kwargs)

        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message):
        def prepare_itlist(msgs):
            content, images, text_content = '', [], ''
            first_text_message = True
            for s in msgs:
                if s['type'] == 'image':
                    images.append(s['value'])
                    content += '<image_placeholder>'
                elif s['type'] == 'text':
                    text_content += s['value']
                    if first_text_message:
                        content += self.cot_prompt.format(s['value'])
                        first_text_message = False
                    else:
                        content += s['value']
            return content, images, text_content
        conversation = []
        if 'role' not in message[0]:
            content, images, text_content = prepare_itlist(message)
            conversation.append(dict(role='User', content=content, images=images))
        else:
            role_map = {'user': 'User', 'assistant': 'Assistant'}
            for msgs in message:
                role = role_map[msgs['role']]
                content, images, text_content = prepare_itlist(msgs['content'])
                conversation.append(dict(role=role, content=content, images=images))
        conversation.append(dict(role='Assistant', content=''))

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.system_prompt,
        )
        prompt = sft_format
        return prompt, text_content

    def batch_generate_inner(self, message, dataset, num_generations):
        prompt, prompt_text = self.prepare_inputs(message)

        image_token_num_per_image = self.kwargs['image_token_num_per_image']
        cfg_weight = self.kwargs['cfg_weight']
        temperature = self.kwargs['temperature']
        img_size = self.kwargs['img_size']
        patch_size = self.kwargs['patch_size']

        prompt_inputs = self.vl_chat_processor.tokenizer(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0).to('cuda')
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0).to('cuda')
        input_embeds = self.model.language_model.get_input_embeddings()(prompt_ids)

        # TODO: if num_generations is too large, we need to split it into multiple batches
        if num_generations > 20:
            total_generations = []
            for i in range(prompt_ids.shape[0] // num_generations):
                current_input_embeds = input_embeds[i * num_generations: (i + 1) * num_generations]
                current_attn_mask = prompt_mask[i * num_generations: (i + 1) * num_generations]
                prompt_completion_ids = self.model.language_model.generate(
                    inputs_embeds=current_input_embeds,
                    attention_mask=current_attn_mask,
                    pad_token_id=self.vl_chat_processor.tokenizer.eos_token_id,
                    bos_token_id=self.vl_chat_processor.tokenizer.bos_token_id,
                    eos_token_id=self.vl_chat_processor.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=True,
                    use_cache=True,
                )
                total_generations.append(prompt_completion_ids)
            prompt_completion_ids = torch.cat(total_generations, dim=0)
        else:  # if num_generations == 1, we directly generate all for the batch data
            prompt_completion_ids = self.model.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=prompt_mask,
                pad_token_id=self.vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=self.vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=self.vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
            )

        prompt_ids = prompt_ids
        completion_ids = prompt_completion_ids

        image_gen_prompt_list = []

        prompt = self.vl_chat_processor.tokenizer.decode(prompt_ids[0].cpu().tolist(), skip_special_tokens=True)
        for i in range(completion_ids.shape[0]):
            answer = self.vl_chat_processor.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
            image_gen_prompt = f"{prompt_text}. {answer}"

            conversation = [
                {
                    "role": "User",
                    "content": image_gen_prompt,
                },
                {"role": "Assistant", "content": ""},
            ]
            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )

            image_gen_prompt_list.append(sft_format)

        prompt_inputs = self.vl_chat_processor.tokenizer(
            text=image_gen_prompt_list,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True,
        )  # {'input_ids', 'attention_mask'}

        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        # attention_mask = torch.ones_like(attention_mask)
        # # add image start token at the end
        image_start_token_id = self.vl_chat_processor.tokenizer.encode(self.vl_chat_processor.image_start_tag)[1]
        prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.size(0), 1), image_start_token_id)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)

        inputs_embeds = self.model.language_model.get_input_embeddings()(prompt_ids)
        pad_input_embeds = self.model.language_model.get_input_embeddings()(
            prompt_ids.new_full((1, 1), self.vl_chat_processor.pad_id)
        )
        total_generated_tokens_img = []

        # Currently only one image generation (since the diversity is low)
        for j in range(inputs_embeds.shape[0] // num_generations):
            # Make cond and uncond inputs embeds and attention mask
            cond_inputs_embeds = inputs_embeds[j * num_generations: (j + 1) * num_generations]
            cond_attention_mask = attention_mask[j * num_generations: (j + 1) * num_generations]
            uncond_inputs_embeds = cond_inputs_embeds.clone()
            uncond_inputs_embeds[:, 1:-1] = pad_input_embeds

            inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
            inputs_embeds_img[1::2] = uncond_inputs_embeds
            attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
            attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])
            # import pdb; pdb.set_trace()

            split_size = 2 * num_generations
            for jj in range(0, inputs_embeds_img.shape[0], split_size):
                start = jj
                end = min(jj + split_size, inputs_embeds_img.shape[0])
                generated_tokens = torch.zeros((
                    (end - start) // 2, image_token_num_per_image), dtype=torch.int64
                ).cuda()
                cur_inputs_embeds_img = inputs_embeds_img[start: end]
                cur_attention_mask_img = attention_mask_img[start: end]

                outputs = None
                for k in range(image_token_num_per_image):
                    outputs = self.model.language_model.model(
                        inputs_embeds=cur_inputs_embeds_img,
                        use_cache=True,
                        past_key_values=None if k == 0 else outputs.past_key_values,
                        attention_mask=cur_attention_mask_img
                    )

                    hidden_states = outputs.last_hidden_state
                    logits = self.model.gen_head(hidden_states[:, -1, :])
                    logit_cond = logits[0::2, :]
                    logit_uncond = logits[1::2, :]

                    logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                    probs = torch.softmax(logits / temperature, dim=-1)

                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens[:, k] = next_token.squeeze(dim=-1)

                    next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                    img_embeds = self.model.prepare_gen_img_embeds(next_token)
                    cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                    cur_attention_mask_img = torch.cat([
                        cur_attention_mask_img,
                        cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.int)
                    ], dim=1)

                total_generated_tokens_img.append(generated_tokens)

        total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[num_generations, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((num_generations, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return [PIL.Image.fromarray(img) for img in visual_img]
