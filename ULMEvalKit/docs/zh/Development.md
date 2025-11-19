# 开发新的 Benchmark / ULM

> 🛠️ 如何在 ULMEvalKit 中实现一个新的 Benchmark / ULM？

## 实现一个新的 benchmark

在 ULMEvalKit 中，benchmark 被组织为数据集类。当您尝试实现一个新的 benchmark 时，您可以复用现有的数据集类（*例如*，在实现新的 T2I benchmark 时可以复用 `TextBaseDataset`），或者支持一个新的数据集类。每个数据集必须具有以下两个成员函数（可以复用父类的函数或实现自己的函数）：

- `build_prompt(self, line)`：函数输入 `line` 是一个整数（样本索引）或一个 `pd.Series` 对象（样本的原始记录）。函数输出一个多模态消息，作为 ULM 的输入。`多模态消息`是一个采用以下格式的交错多模态消息列表（该示例包括一张图片和一条文本消息）：`[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`。
- `evaluate(self, eval_file, **judge_kwargs)`：函数输入 `eval_file` 是 ULM 的预测（通常为 `.pkl` 格式）。如果 benchmark 需要外部 VLM 进行评估，则 `judge_kwargs` 可以传递用于 LLM 的参数。函数以 `dict` 或 `pd.DataFrame` 的形式输出 benchmark 评估结果（指标）。

接下来我们简要介绍在 ULMEvalKit 下实现新 benchmark 的一般步骤：

### 1. 准备您的 benchmark tsv 文件

目前，我们将一个 benchmark 组织为单个 TSV 文件。在推理期间，数据文件将从定义的 `DATASET_URL` 链接自动下载到 `$LMUData` 文件（默认路径为 `$HOME/LMUData`，如果未显式设置）。您可以将准备好的 TSV 文件上传到可下载地址（例如 Huggingface）或通过 <EMAIL> 发送给我们。我们将协助将数据集上传到服务器。您也可以在环境变量中自定义 `LMUData` 路径：`LMUData=/path/to/your/data`。

**`TSV` 文件中必填字段介绍：**

- **index:** 整数，在 `tsv` 中每行唯一
- **prompt:** 字符串，要发送给 ULM 的文本提示
- **image:**（对于 text-to-image 的 benchmark 不必须）输入图像的 base64 编码（用于图像-文本到图像任务），您可以使用 ulmeval/smp/ulm.py 中实现的 API 进行编码和解码：
    - 编码：encode_image_to_base64（用于 PIL Image）/ encode_image_file_to_base64（用于图像文件路径）
    - 解码：decode_base64_to_image（用于 PIL Image）/ decode_base64_to_image_file（用于图像文件路径）

请注意，除了 `index` 字段外，其他字段都可以自定义。您可以根据需要添加其他字段，例如 `answer`、`category`、`difficulty` 等。您也可以将 `prompt` 的名称更改为 `text` 或其他名称。但是您对 `build_prompt(self, line)` 和 `evaluate(self, eval_file, **judge_kwargs)` 的实现应与 `tsv` 文件中的字段名称保持一致。

### 2. 自定义您的 benchmark 提示词

`TextBaseDataset` 和 `ImageBaseDataset` 定义了默认的提示格式。如果您需要添加特定于数据集的提示或以 `Interleave` 格式将输入数据输入模型，您可以通过 `build_prompt(self, line)` 函数实现。该函数以 TSV 文件中的一行作为输入，包含 index、image、question 等字段。该函数返回一个多模态消息字典列表 `msg`，格式为 `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`，包括图像路径和要输入到 ULMs 的文本提示。对于交错类型的输入，您可以直接将图像路径的字典放在图像标记的位置。

下面是一个自定义 `build_prompt` 函数的简单示例：
```python
def build_prompt(self, line):
    if isinstance(line, int):
        line = self.data.iloc[line]
    question = line['prompt']
    msgs = []
    msgs.append(dict(type='text', value=question))
    return msgs
```

### 3. 自定义您的 benchmark 指标

要为新的 benchmark 添加评估，您需要自定义一个类对象来实现数据集的指标计算。多模态数据集继承自 `ulmeval/dataset/text_base.py` 或 `ulmeval/dataset/image_base.py` 中的 `TextBaseDataset` 或 `ImageBaseDataset` 对象。TYPE 定义了数据集的类型，`DATASET_URL` 是数据集的下载地址，`DATASET_MD5` 是用于数据集文件一致性检查的 MD5 校验和。

在这个类中，**您需要实现** `evaluate(eval_file, **judge_kwargs)` 类函数来计算自定义数据集的指标并输出结果。函数输入 `eval_file` 是模型预测结果文件 `{model_name}_{dataset}.pkl` 的路径。该文件可以使用 `load(eval_file)` 方法作为 pandas.DataFrame 读取，包含 index、question、answer、category、prediction 等字段。请注意，只有 `prediction` 字段可以在您运行 `run.py` 时由推理脚本**自动生成**，其他字段来自原始数据集（`tsv` 文件）。`judge_kwargs` 将传递一个与评估相关的字典，例如 `judge model` 的名称、API 请求线程数等。函数的**返回值**是计算出的准确率和其他指标，格式为由列表组成的字典，并组织成 pandas.DataFrame。

## 实现一个新的模型

**支持 `generate_inner` API（强制要求）**

所有现有模型都在 `ulmeval/ulm` 中实现。对于一个最小化的模型，您的模型类**必须实现方法** `generate_inner(msgs, dataset=None)`。在此函数中，您将一个多模态消息输入给 ULM 并返回 ULM 的输出（输出为一个图像 `PIL.Image`）。可选参数 `dataset` 可用作模型在各种推理策略之间切换的标志。您还可以实现 `batch_generation_inner(msgs, dataset, num_generations)` 函数以支持批量推理，该函数返回长度为 `num_generations` 的 `[PIL.Image]` 列表。

多模态消息 `msgs` 是一个字典列表，每个字典有两个键：type 和 value：
- `type`：我们目前支持两种类型，可选值为 ["image", "text"]。
- `value`：当 `type=='text'` 时，value 是文本消息（单个字符串）；当 `type=='image'` 时，value 可以是图像文件的本地路径，或图像 URL。

目前，一个多模态消息可能包含任意交错的图像和文本。如果您的模型不支持这一点，一种做法是取第一张图像和拼接的文本消息作为输入。

以下是一些多模态消息的示例：

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

为了方便起见，我们还支持将字符串列表作为输入。在这种情况下，我们将检查字符串是否是图像路径或图像 URL，并自动将其转换为 `list[dict]` 格式：

```python
from ulmeval.config import supported_ULM
model = supported_ULM['Janus-Pro-1B']()
ret = model.generate(['A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.'], num_generations=2)
ret[0].save('./assets/photo0.png')
ret[1].save('./assets/photo1.png')
```

## 为 ULMEvalKit 做贡献

如果您想向 **ULMEvalKit** 贡献代码，请在提交 PR 之前进行 pre-commit 检查。这有助于保持代码整洁。

```bash
# 在 ULMEvalKit 目录下，安装 pre-commit hook：
pip install pre-commit
pre-commit install
pre-commit run --all-files
# 然后您可以提交您的代码。
```
