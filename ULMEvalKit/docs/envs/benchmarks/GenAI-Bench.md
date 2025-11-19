# GenAI-Bench Environment

```bash
conda create --name genai_eval python=3.10
conda activate genai_eval

cd ULMEvalKit
git clone https://github.com/linzhiqiu/t2v_metrics.git
cd t2v_metrics
pip install -e .

cd ..
pip install -e .
```
