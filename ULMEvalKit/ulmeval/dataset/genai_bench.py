from ..smp import *
import os
from .text_base import TextBaseDataset
try:
    import t2v_metrics.t2v_metrics as t2v_metrics
except:
    logger = get_logger('T2VMetrics')
    logger.critical('T2VMetrics is not installed. Please install it if you want to evaluate GenAI_Bench.')
import numpy as np
import torch.distributed as dist


TAG_GROUPS = {
    'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
    'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced']
}


class GenAI_Bench(TextBaseDataset):
    """GenAI_Bench dataset wrapper for image generation and scoring."""

    TYPE = 'T2I'
    MODALITY = 'TEXT'
    NUM_GENERATIONS = 1

    DATASET_URL = {
        'GenAI_Bench': 'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/GenAI_Bench.tsv',
    }

    DATASET_MD5 = {
        'GenAI_Bench': 'c400de695ecf172df4b92514e5e17049',
    }

    def __init__(self, dataset='GenAI_Bench_Image'):
        self.dataset_name = dataset
        data = self.load_data(dataset)
        # enforce string index for consistent merging
        data['index'] = [str(x) for x in data['index']]
        self.data = data
        self.post_build(dataset)

    def prepare_tsv(self, url, file_md5=None):
        """Load a TSV/CSV path that you already have locally via `load` helper."""
        # In your project, `load` handles .tsv/.csv/.xlsx/.pkl by extension.
        # Here `url` is a local path for WISE_all.
        return load(url)

    def build_prompt(self, line):
        """Build a text-only prompt list."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt_text = line['prompt']
        messages = []
        messages.append(dict(type='text', value=prompt_text))
        return messages

    @staticmethod
    def calculate_category_scores(global_score, index_to_labels, tag_groups=TAG_GROUPS):
        category_results = {}
        all_categories = set()
        for categories in tag_groups.values():
            all_categories.update(categories)

        for category in all_categories:
            scores = []
            for index, score in global_score.items():
                labels = index_to_labels.get(index, [])
                if category in labels:
                    scores.append(score.item())
            category_results[category] = {
                'average': sum(scores) / len(scores) if scores else 0.0,
                'count': len(scores),
                'scores': scores
            }

        all_scores = [score.item() for score in global_score.values()]
        category_results['all'] = {
            'average': sum(all_scores) / len(all_scores) if all_scores else 0.0,
            'count': len(all_scores),
            'scores': all_scores
        }

        return category_results

    @staticmethod
    def merge_category_scores_weighted(all_rank_results):
        merged_results = {}

        all_categories = set()
        for rank_results in all_rank_results:
            all_categories.update(rank_results.keys())

        for category in all_categories:
            total_score = 0.0
            total_count = 0

            for rank_results in all_rank_results:
                if category in rank_results:
                    result = rank_results[category]
                    total_score += result['average'] * result['count']
                    total_count += result['count']

            if total_count > 0:
                merged_results[category] = {
                    'average': total_score / total_count,
                    'count': total_count
                }
            else:
                merged_results[category] = {
                    'average': 0.0,
                    'count': 0
                }

        return merged_results

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        storage_score = eval_file.replace('.pkl', '_score.pkl')

        data = load(eval_file)
        lt = len(data)

        chunk_size = (lt + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, lt)
        data = data.iloc[start_idx:end_idx]
        lines = [data.iloc[i] for i in range(len(data))]

        tups = []
        index_to_labels = {}
        for line in lines:
            for i in range(len(line["prediction"])):
                new_line = line.copy()
                new_line['prediction'] = new_line['prediction'][i]
                tups.append(new_line)
                index_to_labels[new_line['index']] = new_line['labels']

        global_score = []
        clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')
        os.makedirs('genai_bench_generated_images', exist_ok=True)

        global_score = {}
        for tup in tups:
            text = tup['prompt']
            tup['prediction'].save(f"genai_bench_generated_images/{tup['index']}.png")
            score = clip_flant5_score(images=[f"genai_bench_generated_images/{tup['index']}.png"], texts=[text]).cpu()
            global_score[tup['index']] = score

        category_averages = self.calculate_category_scores(global_score, index_to_labels)
        dump(category_averages, storage_score.replace('.pkl', f'_{rank}.pkl'))

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            all_rank_scores = []
            for r in range(world_size):
                chunk_file = storage_score.replace('.pkl', f'_{r}.pkl')
                chunk_data = load(chunk_file)
                all_rank_scores.append(chunk_data)

            merged_scores = self.merge_category_scores_weighted(all_rank_scores)
            dump(merged_scores, storage_score)

            for r in range(world_size):
                chunk_file = storage_score.replace('.pkl', f'_{r}.pkl')
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)

        if world_size > 1:
            dist.barrier()

        return merged_scores
