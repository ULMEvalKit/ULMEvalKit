import os
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
from .base import BaseAPI
import numpy as np

headers = 'Content-Type: application/json'

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1',
}


class GPTIMAGE1(BaseAPI):
    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-image-1',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if 'step' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'yi-vision' in model:
            env_key = os.environ.get('YI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'internvl2-pro' in model:
            env_key = os.environ.get('InternVL2_PRO_KEY', '')
            if key is None:
                key = env_key
        elif 'abab' in model:
            env_key = os.environ.get('MiniMax_API_KEY', '')
            if key is None:
                key = env_key
        elif 'moonshot' in model:
            env_key = os.environ.get('MOONSHOT_API_KEY', '')
            if key is None:
                key = env_key
        elif 'grok' in model:
            env_key = os.environ.get('XAI_API_KEY', '')
            if key is None:
                key = env_key
        elif 'gemini' in model and 'preview' in model:
            # Will only handle preview models
            env_key = os.environ.get('GOOGLE_API_KEY', '')
            if key is None:
                key = env_key
            api_base = "https://generativelanguage.googleapis.com/v1beta"
        elif 'ernie' in model:
            env_key = os.environ.get('BAIDU_API_KEY', '')
            if key is None:
                key = env_key
            api_base = 'https://qianfan.baidubce.com/v2'
            self.baidu_appid = os.environ.get('BAIDU_APP_ID', None)
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                assert isinstance(key, str) and key.startswith('sk-'), (
                    f'Illegal openai_key {key}. '
                    'Please set the environment variable OPENAI_API_KEY to your openai key. '
                )

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout
        self.o1_model = ('o1' in model) or ('o3' in model) or ('o4' in model)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    self.logger.info('Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                self.logger.error('Unknown API Base. ')
                raise NotImplementedError
            if os.environ.get('BOYUE', None):
                self.api_base = os.environ.get('BOYUE_API_BASE')
                self.key = os.environ.get('BOYUE_API_KEY')
        self.client = OpenAI(
            api_key=self.key,
            base_url=self.api_base
        )
        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_inner(self, message, dataset=None, **kwargs):
        prompt = ""
        input_images = []
        for msg in message:
            if msg['type'] == 'text':
                prompt = msg['value']
            elif msg['type'] == 'image':
                image_path = msg['value']
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file {image_path} does not exist.")
                image_base64 = self.encode_image_to_base64(image_path)
                input_images.append(image_base64)

        if not prompt:
            raise ValueError("Text prompt cannot be empty.")

        if input_images:
            image_files = []
            for img_base64 in input_images:
                img_bytes = base64.b64decode(img_base64)
                img_file = BytesIO(img_bytes)
                image_files.append(img_file)
            response = self.client.images.edit(
                model=self.model,
                image=image_files,
                prompt=prompt,
                input_fidelity="high"
            )
            if response.data and len(response.data) > 0:
                b64_image = response.data[0].b64_json
                image_bytes = base64.b64decode(b64_image)
                image = Image.open(BytesIO(image_bytes))
                return 0, image, 'Succeeded! '
            else:
                raise ValueError("No image data returned from the model.")
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        if response.data and len(response.data) > 0:
            b64_image = response.data[0].b64_json
            image_bytes = base64.b64decode(b64_image)
            image = Image.open(BytesIO(image_bytes))
            return 0, image, 'Succeeded! '
        else:
            raise ValueError("No image data returned from the model.")
