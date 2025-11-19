from .text_base import TextBaseDataset
from ..smp import load, dump
from ..utils import track_progress_rich
import pandas as pd
import os.path as osp
import json
import ast

FAIL_MSG = 'Failed to obtain answer via API.'


class GenEvalPP(TextBaseDataset):
    """Geneval++ dataset wrapper for text generation and scoring."""

    TYPE = 'T2I'
    MODALITY = "TEXT"
    NUM_GENERATIONS = 1

    DATASET_URL = {
        'GenEvalPP': 'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/GenEvalPP.tsv',
    }

    DATASET_MD5 = {
        'GenEvalPP': '52907d369795a4ce06fe359174f72acf',
    }

    def __init__(self, dataset='GenEvalPP'):
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
        """Evaluate generated images using GPT-4o based on the Geneval++ protocol."""
        from .utils.judge_util import build_judge

        assert eval_file.endswith('.pkl'), 'data file should be a pkl file'
        judge = judge_kwargs['model']
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

            # Parse results and calculate accuracy
            correct_count = 0
            total_count = len(merged)

            for idx, row in merged.iterrows():
                score_result = score_map.get(str(row['index']), FAIL_MSG)
                if score_result != FAIL_MSG:
                    try:
                        # Extract JSON from response
                        parsed_result = self.extract_json_from_response(score_result)
                        if parsed_result.get('correct', 0) == 1:
                            correct_count += 1
                    except Exception:
                        pass

            accuracy = correct_count / total_count if total_count > 0 else 0
            final_result = {
                'correct': correct_count,
                'total': total_count,
                'accuracy': accuracy
            }

            dump(final_result, tgt_file)
        else:
            final_result = load(tgt_file)

        return final_result

    def prepare_score_prompt(self, item):
        """Build the scoring prompt for GPT evaluation."""
        if isinstance(item, pd.Series):
            item = item.to_dict()

        # Convert string representation of list back to list
        try:
            include_objects = ast.literal_eval(item['include'])
        except:
            include_objects = []

        explanation = self.metadata_to_explanation({
            'include': include_objects
        })

        system_prompt = """You are an expert image evaluator.

Your task is to determine whether the given image faithfully satisfies the visual instruction and the expectation checklist.

Follow these rules strictly:
1. The image must match **all** expectations, including:
- Object classes
- Counts of each object
- Colors of each object
- Spatial position within the image (e.g., "above", "below", based on real pixel position)
- Size and relative scale of objects
2. The image must appear as a **natural, coherent, photo-like single image**.
- Do NOT allow stylized images (e.g., cartoons, sketches, anime).
- Do NOT allow collage-style or multi-panel images. Only one consistent, realistic scene is acceptable.
3. Be very strict and conservative in your judgment.

Return your result as a JSON object using this format:
{
"correct": 1 if the image fully satisfies all expectations, else 0,
"reason": "You may explain in detail what is missing or incorrect"
}"""  # noqa: E501

        from ..smp import encode_image_to_base64
        img_list = item['prediction']
        img = img_list[0] if isinstance(img_list, list) else img_list
        image_b64 = encode_image_to_base64(img)
        messages = [
            {
                "role": "system",
                "value": system_prompt
            },
            {
                "role": "user",
                "type": "text",
                "value": f"Instruction:\n{item['question']}\n\nExpectation checklist:\n{explanation}"
            },
            {
                "role": "user",
                "type": "image",
                "value": f"data:image/jpeg;base64,{image_b64}"
            }
        ]
        return messages

    def metadata_to_explanation(self, metadata):
        """Convert metadata to natural language explanation."""
        parts = []

        def format_item(item):
            obj = item["class"]
            count = item.get("count", 1)
            color = item.get("color", None)
            noun = f"{count} {obj + 's' if count > 1 else obj}"
            desc_parts = []
            if color:
                desc_parts.append(f"{color}-colored")
            if desc_parts:
                noun = f"{' '.join(desc_parts)} {noun}"
            return f"{noun} present in the image"

        for item in metadata.get("include", []):
            parts.append(f"- {format_item(item)}.")

        return "This image should contain:\n" + "\n".join(parts)

    def extract_json_from_response(self, text):
        """Extract JSON object from GPT response."""
        import re
        text = text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("No JSON object found")
