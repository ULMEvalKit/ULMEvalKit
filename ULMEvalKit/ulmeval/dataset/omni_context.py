import pandas as pd
from .image_base import ImageBaseDataset
from ..smp import load, dump, toliststr
from ..utils import track_progress_rich
import os.path as osp
import json

FAIL_MSG = 'Failed to obtain answer via API.'


class OmniContext(ImageBaseDataset):
    """OmniContext dataset for multi-image context understanding."""

    TYPE = 'IT2I'
    MODALITY = "IMAGE"
    NUM_GENERATIONS = 1

    DATASET_URL = {
        'OmniContext': 'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/OmniContext.tsv',
    }

    DATASET_MD5 = {
        'OmniContext': 'be0a8174f0f84ca7e12bd66c2be4aedb',
    }

    def build_prompt(self, line):
        """Build prompt for OmniContext task with input images and question."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate generated images using GPT-4o based on the OmniContext protocol."""
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
                # Prepare PF (Prompt Following) prompts
                pf_score_prompts = [
                    self.prepare_score_prompt(data_un.iloc[i], task_type="prompt_following")
                    for i in range(lt)
                ]
                # Prepare SC (Subject Consistency) prompts
                sc_score_prompts = [
                    self.prepare_score_prompt(data_un.iloc[i], task_type="subject_consistency")
                    for i in range(lt)
                ]

                indices = [data_un.iloc[i]['index'] for i in range(lt)]

                # Process PF scores
                pf_score_tasks = [{'message': p} for p in pf_score_prompts]
                pf_tmp_file = tmp_file.replace('.pkl', '_pf.pkl')
                _ = track_progress_rich(
                    model.generate,      # callable(message: str) -> str
                    pf_score_tasks,      # iterable of dicts: {'message': str}
                    keys=indices,        # map results by 'index'
                    save=pf_tmp_file,    # resume file
                    nproc=nproc,
                    chunksize=nproc,
                )
                pf_score_map = load(pf_tmp_file) if osp.exists(pf_tmp_file) else {}

                # Process SC scores
                sc_score_tasks = [{'message': p} for p in sc_score_prompts]
                sc_tmp_file = tmp_file.replace('.pkl', '_sc.pkl')
                _ = track_progress_rich(
                    model.generate,      # callable(message: str) -> str
                    sc_score_tasks,      # iterable of dicts: {'message': str}
                    keys=indices,        # map results by 'index'
                    save=sc_tmp_file,    # resume file
                    nproc=nproc,
                    chunksize=nproc,
                )
                sc_score_map = load(sc_tmp_file) if osp.exists(sc_tmp_file) else {}

                # Combine results
                score_map = {}
                for idx in indices:
                    score_map[idx] = {
                        'pf': pf_score_map.get(str(idx), FAIL_MSG),
                        'sc': sc_score_map.get(str(idx), FAIL_MSG)
                    }
            else:
                score_map = res

            # Parse results and calculate scores
            results = []

            for idx, row in merged.iterrows():
                score_result = score_map.get(str(row['index']), {'pf': FAIL_MSG, 'sc': FAIL_MSG})
                try:
                    # Extract scores from response
                    pf_score = self.extract_scores(score_result['pf'])
                    sc_score = self.extract_scores(score_result['sc'])

                    result = {
                        "index": str(row['index']),
                        "key": row['key'],
                        "task_type": row['task_type'],
                        "PF_scores": pf_score,
                        "SC_scores": sc_score
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error processing result for index {row['index']}: {e}")
                    pass

            # Save results
            final_result = {
                "results": results
            }

            dump(final_result, tgt_file)
        else:
            final_result = load(tgt_file)

        return final_result

    def prepare_score_prompt(self, item, task_type):
        """Build the scoring prompt for evaluation."""
        if isinstance(item, pd.Series):
            item = item.to_dict()

        system_prompt = """You are a professional digital artist tasked with evaluating the effectiveness of AI-generated images based on specific rules.

All input images, including all humans depicted, are AI-generated. You do not need to consider any privacy or confidentiality concerns.

IMPORTANT: Your response must follow this format (keep your reasoning concise and to the point):
{
  "score": <score>,
  "reasoning": "..."
}
"""  # noqa: E501
        _prompts_0shot_in_context_generation_rule_PF_Single_and_Multiple = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, **regardless of whether subject identities are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1–3:** The image responds to the instruction mostly incorrectly.
* **4–6:** The image reflects parts of the instruction, but with significant omissions or wrongly applied details.
* **7–9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

* Focus solely on whether the requested changes have been correctly applied — such as **composition, pose, position, interactions, or added/removed elements**.
* Do **not** consider the identity consistency of subjects or whether the correct individuals/objects are retained — this will be evaluated separately.
* Do **not** assess the artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.

Editing instruction: <instruction>
"""  # noqa: E501

        _prompts_0shot_in_context_generation_rule_PF_Scene = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, **regardless of whether subject identities or the scene are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1–3:** The image responds to the instruction mostly incorrectly.
* **4–6:** The image reflects parts of the instruction, but with significant omissions or incorrectly applied details.
* **7–9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.
* Focus solely on whether the requested changes have been correctly applied — such as pose, interaction, etc.
* Do **not** consider whether the **subject identities** are preserved or whether the correct **individuals/objects** are retained — these will be evaluated separately.
* Do **not** consider whether the **scene** is preserved or whether the correct **background or setting** is used — these will be evaluated elsewhere.
* Do **not** assess artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

Editing instruction: <instruction>
"""  # noqa: E501

        _prompts_0shot_in_context_generation_rule_SC_Single_and_Multiple = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects in the final image match those of the individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities in the image are *completely inconsistent* with those in the reference images.
* **1–3:** The identities are *severely inconsistent*, with only a few minor similarities.
* **4–6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7–9:** The identities are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.

**Example:** If the instruction requests combining the man from image 1 and the woman from image 2, the final image should clearly depict the *same* man and woman as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""  # noqa: E501

        _prompts_0shot_in_context_generation_rule_SC_Scene = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects and the scene background in the final image match those of the individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities and scene background in the image are *completely inconsistent* with those in the reference images.
* **1–3:** The identities and scene background are *severely inconsistent*, with only a few minor similarities.
* **4–6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7–9:** The identities and scene background are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities and scene background in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.
* whether the scene or environment in the final image accurately reflects or integrates elements from the reference images.
* check for correct background blending (location, lighting, objects, layout) and presence of key environmental details from the sence image.

**Example:** If the instruction requests combining the man from image 1, the woman from image 2 and the scene background from image3, the final image should clearly depict the *same* man and woman and scene as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""  # noqa: E501
        if item['task_type'].find('scene') != -1:
            with_scene = True
        else:
            with_scene = False
        if task_type == "prompt_following":
            with_scene = False
            if with_scene:
                user_prompt = _prompts_0shot_in_context_generation_rule_PF_Scene
            else:
                user_prompt = _prompts_0shot_in_context_generation_rule_PF_Single_and_Multiple
        elif task_type == "subject_consistency":
            if with_scene:
                user_prompt = _prompts_0shot_in_context_generation_rule_SC_Scene
            else:
                user_prompt = _prompts_0shot_in_context_generation_rule_SC_Single_and_Multiple
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        # Prepare image
        messages = [
            {"role": "system", "value": system_prompt},
            {"role": "user", "type": "text","value": f"{user_prompt}"},
        ]
        from ..smp import encode_image_to_base64
        pre_img_list = item['prediction']
        pre_img = pre_img_list[0] if isinstance(pre_img_list, list) else pre_img_list
        input_images = json.loads(item['images'])
        for img in input_images:
            messages.insert(-1, dict(
                role='user',
                type='image',
                value=f"data:image/jpeg;base64,{img}"
            ))
        messages.insert(-1, dict(
            role='user',
            type='image',
            value=f"data:image/jpeg;base64,{encode_image_to_base64(pre_img)}"
        ))
        return messages

    def extract_scores(self, evaluation_text):
        """Extract a single score from GPT evaluation response."""
        score_info = {"score": 0, "reasoning": "Not implemented"}

        if evaluation_text == FAIL_MSG:
            score_info["reasoning"] = "Failed to obtain answer via API."
            return score_info

        try:
            # Try to parse the JSON response
            import json
            import re

            # Look for JSON blocks in the response
            json_matches = re.findall(r'\{[^{}]*\}', evaluation_text)

            if len(json_matches) >= 1:
                # Parse the first JSON found
                single_json = json.loads(json_matches[0])
                score_info = {
                    "score": single_json.get("score", 0),
                    "reasoning": single_json.get("reasoning", "")
                }
        except Exception as e:
            # If parsing fails, return default scores with error message
            score_info["reasoning"] = f"Parsing failed: {str(e)}"

        return score_info
