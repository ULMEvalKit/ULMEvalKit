from ..smp import *
from .base import BaseAPI
from io import BytesIO
import base64

headers = 'Content-Type: application/json'


class GeminiFlashGenerationWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gemini-2.0-flash-preview-image-generation',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 project_id='UlmEvalKit',
                 thinking_budget: int = None,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        # for image, high and medium resolution is 258 tokens per image [default], low resolution is 66 tokens per image
        if key is None:
            key = os.environ.get('GOOGLE_API_KEY', None)

        # Try to load backend from environment variable
        assert key is not None  # Vertex does not require API Key
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "Could not import 'google.genai'. Please install it with:\n"
                "    pip install --upgrade google-genai"
            ) from e
        self.genai = genai
        self.client = genai.Client(api_key=key)

        self.project_id = project_id
        self.api_key = key

        if proxy is not None:
            proxy_set(proxy)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def build_msgs_genai(self, inputs):
        text_and_images = [] if self.system_prompt is None else [self.system_prompt]

        for inp in inputs:
            if inp['type'] == 'text':
                text_and_images.append(inp['value'])
            elif inp['type'] == 'image':
                text_and_images.append(Image.open(inp['value']))

        return text_and_images

    def generate_inner(self, inputs, **kwargs) -> str:
        from google.genai import types
        assert isinstance(inputs, list)
        model = self.model
        messages = self.build_msgs_genai(inputs)

        # Configure generation parameters
        config_args = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "response_modalities": ['TEXT', 'IMAGE']
        }

        # If thinking_budget is specified, add thinking_config
        # By default, Gemini 2.5 Pro will automatically select
        # a thinking budget not exceeding 8192 if not specified.
        if self.thinking_budget is not None:
            config_args["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        config_args.update(kwargs)

        try:
            resp = self.client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(**config_args)
            )

            image = None
            for part in resp.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    break
            return 0, image, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')

            return -1, '', ''


class GeminiFlashGeneration(GeminiFlashGenerationWrapper):

    def generate(self, message, dataset=None):
        return super(GeminiFlashGeneration, self).generate(message)


class ImagenWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'imagen-3.0-generate-002',
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 proxy: str = None,
                 project_id='UlmEvalKit',
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        # for image, high and medium resolution is 258 tokens per image [default], low resolution is 66 tokens per image
        if key is None:
            key = os.environ.get('GOOGLE_API_KEY', None)

        # Try to load backend from environment variable
        assert key is not None  # Vertex does not require API Key
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "Could not import 'google.genai'. Please install it with:\n"
                "    pip install --upgrade google-genai"
            ) from e
        self.genai = genai
        self.client = genai.Client(api_key=key)

        self.project_id = project_id
        self.api_key = key

        if proxy is not None:
            proxy_set(proxy)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def build_msgs_genai(self, inputs):
        text = "" if self.system_prompt is None else self.system_prompt

        for inp in inputs:
            if inp['type'] == 'text':
                text += inp['value']
            elif inp['type'] == 'image':
                warnings.warn(
                    "Imagen API does not support image input, only text input is allowed. Ignoring image input."
                )

        return text

    def generate_inner(self, inputs, **kwargs) -> str:
        from google.genai import types
        assert isinstance(inputs, list)
        model = self.model
        messages = self.build_msgs_genai(inputs)

        # Configure generation parameters
        config_args = {
            "number_of_images": 1
        }

        config_args.update(kwargs)
        number_of_images = kwargs.get('number_of_images', 1)
        assert number_of_images == 1, "Our Imagen API only supports generating one image at a time."

        try:
            resp = self.client.models.generate_images(
                model=model,
                prompt=messages,
                config=types.GenerateImagesConfig(**config_args)
            )

            image = None
            for generated_image in resp.generated_images:
                image = generated_image
                break

            return 0, image, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')

            return -1, '', ''


class Imagen(ImagenWrapper):

    def generate(self, message, dataset=None):
        return super(Imagen, self).generate(message)
