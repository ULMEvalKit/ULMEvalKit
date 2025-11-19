import pandas as pd
from .text_base import TextBaseDataset
import torch.distributed as dist
from abc import abstractmethod
from ..smp import *
from ..utils import track_progress_rich
from .utils import build_judge
from collections import defaultdict


class DPGBench(TextBaseDataset):
    TYPE = 'T2I'
    MODALITY = 'TEXT'

    DATASET_URL = {
        'DPGBench': 'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/DPG_Bench.tsv',
    }

    DATASET_MD5 = {
        'DPGBench': 'c1551e34ea5477eaee81153eae19c8bc',
    }

    def __init__(self, dataset='DPGBench', skip_noimg=True):
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

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['text']

        msgs = []
        msgs.append(dict(type='text', value=question))

        return msgs

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        # from .utils.dpgbench import *
        # print('eval_file: ', eval_file)
        rank, world_size = get_rank_and_world_size()
        storage_eval = eval_file.replace('.pkl', '_eval.csv')
        storage_score = eval_file.replace('.pkl', '_score.pkl')

        data = load(eval_file)
        lt = len(data)

        chunk_size = (lt + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, lt)
        data = data.iloc[start_idx:end_idx]
        lines = [data.iloc[i] for i in range(len(data))]
        tups = []

        for line in lines:
            for i in range(len(line["prediction"])):
                new_line = line.copy()
                new_line['prediction'] = new_line['prediction'][i]
                tups.append(new_line)

        global_score = []
        L1_score = defaultdict(list)
        L2_score = defaultdict(list)
        mplug_model = build_judge(**{'model': 'mplug', **judge_kwargs})

        for tup in tups:
            questions = json.loads(tup['questions'])
            scores = dict()
            image = tup['prediction']
            for question in questions:
                proposition_id = question['proposition_id']
                dependency = question['dependency']
                category_broad = question['category_broad']
                category_detailed = question['category_detailed']
                text = question['question_natural_language']
                answer = mplug_model.vqa(image, text)
                validity = (answer == 'yes')
                with open(storage_eval.replace('.csv', f'_{rank}.csv'), 'a') as f:
                    f.write(str(proposition_id) + ', ' + text + ', ' + answer + '\n')

                for parent in dependency:
                    if parent == 0:
                        continue
                    if scores.get(parent, 1.0) == 0.0:
                        validity = False
                        break
                score = 1.0 if validity else 0.0
                scores[proposition_id] = score
                global_score.append(score)
                L1_score[category_broad].append(score)
                L2_score[category_broad + '-' + category_detailed].append(score)

        data = {
            'global_score': global_score,
            'L1_score': dict(L1_score),
            'L2_score': dict(L2_score)
        }
        dump(data, storage_score.replace('.pkl', f'_{rank}.pkl'))

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            merged_global = []
            merged_L1 = defaultdict(list)
            merged_L2 = defaultdict(list)

            for r in range(world_size):
                chunk_file = storage_score.replace('.pkl', f'_{r}.pkl')
                chunk_data = load(chunk_file)
                global_score = chunk_data['global_score']
                L1_score = chunk_data['L1_score']
                L2_score = chunk_data['L2_score']
                merged_global.extend(global_score)
                for k, v in L1_score.items():
                    merged_L1[k].extend(v)
                for k, v in L2_score.items():
                    merged_L2[k].extend(v)

            global_score = sum(merged_global) / len(merged_global)
            L1_score = {k: sum(v) / len(v) for k, v in merged_L1.items()}
            L2_score = {k: sum(v) / len(v) for k, v in merged_L2.items()}
            score = {
                'global_score': global_score,
                'L1_score': L1_score,
                'L2_score': L2_score
            }
            dump(score, storage_score)

        if world_size > 1:
            dist.barrier()

        return score
