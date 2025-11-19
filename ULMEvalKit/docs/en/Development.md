# Develop new Benchmark / ULM

>  🛠️ How to implement a new Benchmark / ULM in ULMEvalKit?

## Implement a new benchmark

In ULMEvalKit, benchmarks are organized as dataset classes. When you try to implement a new benchmark, you can either reuse existing dataset classes (*e.g.*, You can reuse `TextBaseDataset` when implementing a new T2I benchmark), or support a new dataset class. Each dataset must have the following two member functions (either reuse the one of the parent class or implement your own):

- `build_prompt(self, line)`: The function input `line` is an integer (the sample index) or a `pd.Series` object (the raw record of the sample). The function outputs a multi-modal message, serving as the input of an ULM. The `multi-modal message` is an interleaved list of multi-modal messages adopting the following format (the example includes an image and a text message): `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`.
- `evaluate(self, eval_file,  **judge_kwargs)`: The function input `eval_file` is the ULM prediction (typically in `.pkl` format). If the benchmark requires an external VLM for evaluation, then `judge_kwargs` can pass the arguments for the LLM. The function outputs the benchmark evaluation results (metrics) in the form of `dict` or `pd.DataFrame`.

We then brief the typical steps to implement a new benchmark under ULMEvalKit:

### 1. Prepare your benchmark tsv file

Currently, we organize a benchmark as one single TSV file. During inference, the data file will be automatically downloaded from the definited `DATASET_URL` link to `$LMUData` file (default path is `$HOME/LMUData`, if not set explicitly). You can upload the prepared TSV file to a downloadable address (e.g., Huggingface) or send it to us at <EMAIL>. We will assist in uploading the dataset to the server. You can also customize `LMUData` path in the environment variable `LMUData=/path/to/your/data`.

**Intro to mandatory fields in the `TSV` file:**

- **index:** Integer, Unique for each line in `tsv`
- **prompt:** String, The text prompt to be sent to the ULM
- **image:** (for text-to-image benchmark not mandatory) The base64 of the input image (for image-text-to-image tasks), you can use APIs implemented in ulmeval/smp/ulm.py for encoding and decoding:
    - Encoding: encode_image_to_base64 (for PIL Image) / encode_image_file_to_base64 (for image file path)
    - Decoding: decode_base64_to_image(for PIL Image) / decode_base64_to_image_file (for image file path)

Note that except for the `index` field, other fields can be customized. You can add other fields as needed, such as `answer`, `category`, `difficulty`, etc. You can also change the name of the `prompt` to `text` or other names. But your implementation of `build_prompt(self, line)` and `evaluate(self, eval_file, **judge_kwargs)` should be consistent with the field names in the `tsv` file.

### 2. Cutomize your benchmark prompt

`TextBaseDataset` and `ImageBaseDataset` defines the default prompt format. If you need to add prompts specific to the dataset or input data in the `Interleave` format to the model, you can implement this through the `build_prompt(self, line)` function. This function takes a line from a TSV file as input, containing fields such as index, image, question, etc. The function returns a dictionary list of multimodal messages `msg` in the format `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`, including the image path and the text prompt to be input into ULMs. For interleave type inputs, you can directly place the dictionary of the image path at the image token position.

Below is a simple example of a custom `build_prompt` function:
```python
def build_prompt(self, line):
    if isinstance(line, int):
        line = self.data.iloc[line]
    question = line['prompt']
    msgs = []
    msgs.append(dict(type='text', value=question))
    return msgs
```

### 3. Cutomize your benchmark metrics

To add evaluation for a new benchmark, you need to customize a class object to implement the dataset’s metrics calculation. Multimodal datasets inherit from the `TextBaseDataset` or `ImageBaseDataset` object in `ulmeval/dataset/text_base.py` or `ulmeval/dataset/image_base.py`. The TYPE defines the type of dataset, `DATASET_URL` is the download address of the dataset, and `DATASET_MD5` is the MD5 checksum for consistency checking of the dataset file.

In this class, **you need to implement** the `evaluate(eval_file, **judge_kwargs)` class function to calculate metrics and output results for the custom dataset. The function input `eval_file` is the path to the model prediction results file `{model_name}_{dataset}.pkl`. This file can be read as a pandas.DataFrame using the `load(eval_file)` method, containing fields such as index, question, answer, category, prediction, etc. Note that only the `prediction` field can be **automatically generated** by the inference script when you run `run.py`, and other fields are from the original dataset (`tsv` file). The `judge_kwargs` will pass a dictionary related to evaluation, such as the name of the `judge model`, the number of API request threads, etc. **The return value** of the function is the calculated accuracy and other metrics, formatted as a dictionary composed of lists, organized into a pandas.DataFrame.

## Implement a new model

**Support `generate_inner` API (mandatory).**

All existing models are implemented in `ulmeval/ulm`. For a minimal model, your model class **must implement the method** `generate_inner(msgs, dataset=None)`. In this function, you feed a multi-modal message to your ULM and return the ULM prediction (which is an image `PIL.Image`). The optional argument `dataset` can be used as the flag for the model to switch among various inference strategies. You can also implement the `batch_generation_inner(msgs, dataset, num_generations)` function to support batch inference, which returns `[PIL.Image]` of length `num_generations`.

The multi-modal messages `msgs` is a list of dictionaries, each dictionary has two keys: type and value:
- `type`: We currently support two types, choices are ["image", "text"].
- `value`: When `type=='text'`, the value is the text message (a single string); when `type=='image'`, the value can be the local path of an image file, or the image URL.

Currently a multi-modal message may contain arbitrarily interleaved images and texts. If your model do not support that, a practice can be taking the 1st image and concatenated text messages as the input.

Here are some examples of multi-modal messages:

```python
from ulmeval.config import supported_ULM
model = supported_ULM['Janus-Pro-1B']()
IMAGE_PTH = 'assets/apple.jpg'
msg = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='Add another apple to the image')
]
ret = model.generate(msg)[0]
ret.save('./assets/apple2.png')
```

For convenience sake, we also support to take a list of string as inputs. In that case, we will check if a string is an image path or image URL and automatically convert it to the `list[dict]` format:

```python
from ulmeval.config import supported_ULM
model = supported_ULM['Janus-Pro-1B']()
ret = model.generate(['A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.'], num_generations=2)
ret[0].save('./assets/photo0.png')
ret[1].save('./assets/photo1.png')
```

## Contribute to ULMEvalKit

If you want to contribute codes to **ULMEvalKit**, please do the pre-commit check before you submit a PR. That helps to keep the code tidy.

```bash
# Under the directory of ULMEvalKit, install the pre-commit hook:
pip install pre-commit
pre-commit install
pre-commit run --all-files
# Then you can commit your code.
```
