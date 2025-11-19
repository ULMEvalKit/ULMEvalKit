import pandas as pd
from .text_base import TextBaseDataset
import torch.distributed as dist
from abc import abstractmethod
from ..smp import *
from ..utils import track_progress_rich
from .utils import build_judge


class GenEval(TextBaseDataset):

    TYPE = 'T2I'
    MODALITY = "TEXT"
    NUM_GENERATIONS = 4
    DATASET_URL = {
        'GenEval': 'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/GenEval.tsv',
    }

    DATASET_MD5 = {
        'GenEval': '2c3a418e2b5a53a2d411b050aad834b4',
    }

    def __init__(self, dataset='GenEval', skip_noimg=True):
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset

        data = self.load_data(dataset)
        data['index'] = [int(x) for x in data['index']]

        self.meta_only = True

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question']
        return [question]

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.geneval import GenEval_auxeval_score, GenEval_acc

        rank, world_size = get_rank_and_world_size()
        storage_eval = eval_file.replace('.pkl', '_eval.pkl')
        storage_score = eval_file.replace('.pkl', '_score.csv')
        nproc = judge_kwargs.pop('nproc', 4)
        data = load(eval_file)
        data = data.explode("prediction").reset_index(drop=True)
        det_model = build_judge(**{'model': 'mask2former', **judge_kwargs})
        clip_model = build_judge(**{'model': 'openclip-vit-l-14', **judge_kwargs})

        # split data into chunks
        lt = len(data)
        chunk_size = (lt + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, lt)
        data = data.iloc[start_idx:end_idx]
        tups = [(det_model, clip_model, data.iloc[i]) for i in range(len(data))]
        indices = list(range(len(tups)))

        if len(indices):
            ans = track_progress_rich(
                GenEval_auxeval_score,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=indices,
            )
        data['correct'] = [ans[idx]['correct'] for idx in range(len(data['index']))]
        data['reason'] = [
            ans[idx]['reason'] for idx in range(len(data['index']))
        ]
        data['details'] = [
            ans[idx]['details'] for idx in range(len(data['index']))
        ]
        dump(data, storage_eval.replace('.pkl', f'_{rank}.pkl'))

        score = None
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            chunks = []
            for r in range(world_size):
                chunk_file = storage_eval.replace('.pkl', f'_{r}.pkl')
                chunk_data = load(chunk_file)
                chunks.append(chunk_data)
                os.remove(chunk_file)
            full_data = pd.concat(chunks)
            full_data = full_data.sort_values(by='index')
            dump(full_data, storage_eval)
            score = GenEval_acc(storage_eval)
            dump(score, storage_score)
        if world_size > 1:
            dist.barrier()
        return score
