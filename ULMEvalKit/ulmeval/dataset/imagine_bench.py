import pandas as pd
from .text_base import TextBaseDataset
from ..smp import load, dump
from ..utils import track_progress_rich
import os.path as osp
import re

FAIL_MSG = 'Failed to obtain answer via API.'


class ImagineBench(TextBaseDataset):
    """ImagineBench dataset wrapper for text generation and scoring."""

    TYPE = 'T2I'
    MODALITY = "TEXT"
    NUM_GENERATIONS = 1

    DATASET_URL = {
        'ImagineBench': 'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/ImagineBench.tsv',
    }

    DATASET_MD5 = {
        'ImagineBench': 'dd65cfbb5672946a439b16b7696af2a0',
    }

    def __init__(self, dataset='ImagineBench'):
        self.dataset_name = dataset
        data = self.load_data(dataset)
        # enforce string index for consistent merging
        data['index'] = [str(x) for x in data['index']]
        self.data = data
        self.post_build(dataset)

    def build_prompt(self, line):
        """Build a text-only prompt list for image generation."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        prompt_text = line['question']
        messages = []
        messages.append(dict(type='text', value=prompt_text))
        return messages

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate generated images using GPT-4o based on the ImagineBench protocol."""
        from .utils.judge_util import build_judge

        assert eval_file.endswith('.pkl'), 'data file should be a pkl file'
        judge = judge_kwargs.get('model', None)
        if judge is None:
            raise ValueError("Missing 'model' key in judge_kwargs. Please specify a judge model.")

        nproc = judge_kwargs.pop('nproc', 4)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)

        tmp_file = eval_file.replace('.pkl', f'_{judge}_tmp.pkl')
        tgt_file = eval_file.replace('.pkl', f'_{judge}_rating.json')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)

        eval_df = load(eval_file)
        meta_df = self.data.copy()

        # unify index dtype
        if meta_df['index'].dtype != eval_df['index'].dtype:
            eval_df['index'] = eval_df['index'].astype(str)
            meta_df['index'] = meta_df['index'].astype(str)

        dup_cols = (set(meta_df.columns) & set(eval_df.columns)) - {'index'}
        if dup_cols:
            meta_df = meta_df.drop(columns=list(dup_cols), errors='ignore')

        merged = pd.merge(eval_df, meta_df, on='index', how='inner')
        # run scoring if not already present
        if not osp.exists(tgt_file):
            # resume support: load tmp results and drop failures
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            todo_mask = ~merged['index'].isin(res.keys())
            data_un = merged[todo_mask].reset_index(drop=True)

            lt = len(data_un)
            # Build plain string prompts to avoid BaseAPI 'value' key issues
            if lt > 0:
                score_prompts = [self.prepare_score_prompt(data_un.iloc[i]) for i in range(lt)]
                indices = [data_un.iloc[i]['index'] for i in range(lt)]
                score_tasks = [{'message': p} for p in score_prompts]
                _ = track_progress_rich(
                    model.generate,      # callable(message: str) -> str
                    score_tasks,         # iterable of dicts: {'message': str}
                    keys=indices,        # map results by 'index'
                    save=tmp_file,       # resume file
                    nproc=nproc,
                    chunksize=nproc,
                )
                new_map = load(tmp_file)
                score_map = {**res, **new_map}
            else:
                score_map = res

            # Parse results and calculate scores
            results = []
            type_scores = {}

            for idx, row in merged.iterrows():
                score_result = score_map.get(str(row['index']), FAIL_MSG)
                if score_result != FAIL_MSG:
                    try:
                        # Extract scores from response
                        scores = self.extract_scores(score_result)
                        weighted_score = (
                            0.6 * scores["Fantasy Fulfillment"]
                            + 0.3 * scores["Identity Preservation"]
                            + 0.1 * scores["Aesthetic Quality"]
                        )

                        result = {
                            "index": str(row['index']),
                            "id": row['id'],
                            "type": row['type'],
                            "Fantasy Fulfillment": scores["Fantasy Fulfillment"],
                            "Identity Preservation": scores["Identity Preservation"],
                            "Aesthetic Quality": scores["Aesthetic Quality"],
                            "Weighted Score": round(weighted_score, 3)
                        }
                        results.append(result)

                        # Group by type for statistics
                        item_type = row['type']
                        if item_type not in type_scores:
                            type_scores[item_type] = {
                                "Fantasy Fulfillment": [],
                                "Identity Preservation": [],
                                "Aesthetic Quality": [],
                                "Weighted Score": []
                            }
                        type_scores[item_type]["Fantasy Fulfillment"].append(scores["Fantasy Fulfillment"])
                        type_scores[item_type]["Identity Preservation"].append(scores["Identity Preservation"])
                        type_scores[item_type]["Aesthetic Quality"].append(scores["Aesthetic Quality"])
                        type_scores[item_type]["Weighted Score"].append(weighted_score)
                    except Exception as e:
                        print(f"Error processing result for index {row['index']}: {e}")
                        pass

            # Calculate statistics by type
            statistics = {}
            for item_type, scores in type_scores.items():
                statistics[item_type] = {
                    "Fantasy Fulfillment": round(sum(scores["Fantasy Fulfillment"]) / len(scores["Fantasy Fulfillment"]), 3),  # noqa: E501
                    "Identity Preservation": round(sum(scores["Identity Preservation"]) / len(scores["Identity Preservation"]), 3),  # noqa: E501
                    "Aesthetic Quality": round(sum(scores["Aesthetic Quality"]) / len(scores["Aesthetic Quality"]), 3),
                    "Weighted Score": round(sum(scores["Weighted Score"]) / len(scores["Weighted Score"]), 3),
                    "count": len(scores["Weighted Score"])
                }  # noqa: E501

            # Overall statistics
            all_weighted_scores = [r["Weighted Score"] for r in results]
            overall_statistics = {
                "overall": {
                    "Weighted Score": round(sum(all_weighted_scores) / len(all_weighted_scores), 3),
                    "count": len(all_weighted_scores)
                }
            }

            final_result = {
                "results": results,
                "statistics_by_type": statistics,
                "overall_statistics": overall_statistics
            }

            dump(final_result, tgt_file)
        else:
            final_result = load(tgt_file)

        return final_result

    def prepare_score_prompt(self, item):
        """Build the scoring prompt for GPT evaluation."""
        if isinstance(item, pd.Series):
            item = item.to_dict()

        system_prompt = """You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores. Your job is to evaluate how well the image fulfills the fantasy design task, considering all instructions with maximum precision."""  # noqa: E501

        user_prompt = f"""Please evaluate strictly and return ONLY the three scores as requested.

# Fantasy Object Image Evaluation Protocol (Human-based tasks)

## Input Parameters
- PROMPT: [Original instruction provided to the model]
- EXPLANATION: [Detailed explanation of what the prompt is trying to achieve]

---

## Scoring Criteria (0–10 for each)

**Fantasy Fulfillment (0–10):**
How well does the image realize the intended fantasy transformation described in the prompt?
- 0: No sign of the transformation; the fantasy idea is entirely ignored or contradicted.
- 1–3: The transformation is misunderstood or poorly executed, with key fantasy features missing, wrong, or distorted.
- 4–6: Some aspects of the transformation appear, but with clear flaws — such as vague, generic, or misaligned features.
- 7–9: Most fantasy elements are present and understandable, but minor details may be off in material, form, or integration.
- 10: The transformation is fully and precisely implemented. Every visual element aligns with the prompt's intent — including texture, shape, integration, and plausibility.

To score 10, the image must exactly reflect the imagined transformation with no major deviations.

**Identity Preservation (0–10):**
How clearly does the image preserve the recognizable identity of the original object/person despite the fantasy alteration?
- 0: The object/person is completely unrecognizable or heavily distorted.
- 1–3: The identity is barely preserved; key visual traits are missing or incorrect.
- 4–6: Some identity traits remain, but many are altered, stylized, or inconsistent.
- 7–9: The core features are retained well, with minor issues.
- 10: The base object/person is clearly and faithfully represented in all major aspects.
Stylized or cartoon-like rendering should be rated **lower**, even if the shape is roughly preserved. Identity must be preserved in **realistic detail**, not just symbolic outline.

**Aesthetic Quality (0–10):**
How visually appealing, clear, and realistic is the image overall?
- 0: Poor quality, low resolution, or visually broken.
- 1–3: Basic rendering flaws or artifacts significantly hurt visual quality.
- 4–6: Adequate quality with moderate imperfections.
- 7–9: High-quality rendering with good composition and polish.
- 10: Excellent visual clarity, realism, and artistic balance.

---

## Output Format
Return scores (0–10) and **brief justification** for each item.

Output Format Example:

Fantasy Fulfillment: <score>
Reason: <one-sentence explanation>

Identity Preservation: <score>
Reason: <one-sentence explanation>

Aesthetic Quality: <score>
Reason: <one-sentence explanation>

---

Only provide the scores and reasons. Do not include any extra formatting or comments.

---

## Enforcement Notes
- Be extremely strict and objective.
- A score of **10** must indicate complete success and flawless execution.
- If the fantasy transformation is weak or confusing → downgrade **Fantasy Fulfillment**.
- If the base object/person is unrecognizable or overly stylized → downgrade **Identity Preservation**.
- If realism or visual appeal is compromised → downgrade **Aesthetic Quality**.
- Reject images that exhibit cartoonish rendering or inconsistent fantasy logic.

---

Here are the inputs for evaluation:
PROMPT: "{item['question']}"
EXPLANATION: "{item['note']}"

Please evaluate this image:
"""  # noqa: E501
        from ..smp import encode_image_to_base64
        img_list = item['prediction']
        img = img_list[0] if isinstance(img_list, list) else img_list
        image_b64 = encode_image_to_base64(img)
        messages = [
            {"role": "system", "value": system_prompt},
            {"role": "user", "type": "text","value": f"{user_prompt}"},
            {"role": "user", "type": "image","value": f"data:image/jpeg;base64,{image_b64}"}
        ]
        return messages

    def extract_scores(self, evaluation_text):
        """Extract scores from GPT evaluation response."""
        pattern = r"(Fantasy Fulfillment|Identity Preservation|Aesthetic Quality)\s*[:：]?\s*(\d{1,2})"
        scores = {
            "Fantasy Fulfillment": 0,
            "Identity Preservation": 0,
            "Aesthetic Quality": 0
        }
        for match in re.findall(pattern, evaluation_text):
            field, value = match
            field = field.strip()
            try:
                val = float(value)
                val = max(0, min(val, 10))
                if field in scores:
                    scores[field] = val
            except ValueError:
                continue
        return scores
