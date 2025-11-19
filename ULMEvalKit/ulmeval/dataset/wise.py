import os
import os.path as osp
import warnings
import pandas as pd
from .text_base import TextBaseDataset
from ulmeval.smp import load, dump  # adjust import path if your project layout differs
from ulmeval.dataset.utils.wise import (
    prepare_response_prompt,
    prepare_score_prompt,
    get_dimension_rating,
)
from ulmeval.utils.mp_util import track_progress_rich  # adjust path if needed
from ulmeval.dataset.utils.judge_util import build_judge  # your existing judge builder

FAIL_MSG = 'Failed to obtain answer via API.'


class WISE(TextBaseDataset):
    """WISE dataset wrapper for image generation and scoring."""

    TYPE = 'T2I'
    MODALITY = 'TEXT'
    NUM_GENERATIONS = 1

    DATASET_URL = {
        'WISE_all': 'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/WISE_all.tsv',
    }

    DATASET_MD5 = {
        'WISE_all': 'f4fb0fd05e83bd1c5ec48a37abe91735',
    }

    def __init__(self, dataset='WISE_all'):
        self.dataset_name = dataset
        data = self.load_data(dataset)
        # enforce string index for consistent merging
        data['index'] = [str(x) for x in data['index']]
        self.data = data
        self.mode = 0
        if dataset == 'WSIE_all':
            self.mode = 1
        self.post_build(dataset)

    def build_prompt(self, line):
        """Build a text-only prompt list for JanusGeneration.generate_inner."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt_text = line['prompt']
        messages = []
        messages.append(dict(type='text', value=prompt_text))
        return messages

    def evaluate(self, eval_file, **judge_kwargs):
        """Score generated images with an LLM and aggregate WISE metrics.

        eval_file: .pkl with columns at least ['index', 'prediction'] where
                   'prediction' is a PIL.Image.Image.
        """
        assert eval_file.endswith('.pkl'), 'data file should be a pkl file'
        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 4)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)
        tmp_file = eval_file.replace('.pkl', f'_{judge}_tmp.pkl')
        score_file = eval_file.replace('.pkl', f'_{judge}_score.xlsx')
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

        if not osp.exists(score_file):
            # resume support: load tmp results and drop failures
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            todo_mask = ~merged['index'].isin(res.keys())
            data_un = merged[todo_mask].reset_index(drop=True)

            lt = len(data_un)
            # Build plain string prompts to avoid BaseAPI 'value' key issues
            if lt > 0:
                score_prompts = [prepare_score_prompt(data_un.iloc[i]) for i in range(lt)]
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
            merged['score'] = [score_map.get(idx, FAIL_MSG) for idx in merged['index']]
            dump(merged, score_file)

        rating = get_dimension_rating(self.mode, score_file)
        dump(rating, tgt_file)
        return rating
