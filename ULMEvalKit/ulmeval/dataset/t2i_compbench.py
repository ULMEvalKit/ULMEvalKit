import torch.distributed as dist
from .text_base import TextBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *


class T2ICompBench(TextBaseDataset):
    TYPE = 'T2I'
    MODALITY = 'TEXT'
    NUM_GENERATIONS = 10

    DATASET_URL = {
        'T2ICompBench_VAL':
        'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/T2ICompBench_VAL.tsv',
        'T2ICompBench_Color_VAL':
        'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/T2ICompBench_Color_VAL.tsv',
        'T2ICompBench_Shape_VAL':
        'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/T2ICompBench_Shape_VAL.tsv',
        'T2ICompBench_Texture_VAL':
        'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/T2ICompBench_Texture_VAL.tsv',
        'T2ICompBench_Spatial_VAL':
        'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/T2ICompBench_Spatial_VAL.tsv',
        'T2ICompBench_non_Spatial_VAL':
        'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/T2ICompBench_non_Spatial_VAL.tsv',
        'T2ICompBench_Complex_VAL':
        'https://huggingface.co/datasets/CaraJ/ULMEvalKit/resolve/main/T2ICompBench_Complex_VAL.tsv',
    }

    DATASET_MD5 = {
        'T2ICompBench_VAL':
        'b46fb37aa7d973f661cd7b43aa2e58b8',
        'T2ICompBench_Color_VAL':
        'e718d1a73cc793a72f287c91e34e17d7',
        'T2ICompBench_Shape_VAL':
        '93669f86365e0565985036e6e8c6b9fe',
        'T2ICompBench_Texture_VAL':
        'b9c7821e7091f6f7229e2276ea2da0a2',
        'T2ICompBench_Spatial_VAL':
        '6e301c6d32bf0465ef89a53353a3c711',
        'T2ICompBench_non_Spatial_VAL':
        '8cb88471b72347196e448de9f36ae760',
        'T2ICompBench_Complex_VAL':
        'dbe4bee11bc00258daf5d9666127b557',
    }

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        msgs = []
        msgs.append(dict(type='text', value=question))

        return msgs

    def save_temp_images(self, data, save_dir):
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        for line in lines:
            if isinstance(line['prediction'], list):
                for idx, img in enumerate(line['prediction']):
                    img_name = line['question'] + f'_{idx:0>6}' + '.jpg'
                    img.save(os.path.join(save_dir, img_name))
            elif isinstance(line['prediction'], Image.Image):
                img_name = line['question'] + f'_{0:0>6}' + '.jpg'
                line['prediction'].save(os.path.join(save_dir, img_name))
        return

    def remove_temp_images(self, save_dir):
        for img in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, img))
        os.removedirs(save_dir)
        return

    def get_tasks_for_rank(self, tasks, rank=0, world_size=1):
        num_tasks = len(tasks)
        if world_size == 1:
            return tasks
        elif (num_tasks % world_size) != 0:
            if rank == 0:
                print(f"Num tasks {num_tasks} not divisible by world_size {world_size}, use serial mode")
                return tasks
            else:
                return []
        else:
            if num_tasks >= world_size:
                tasks_per_rank = num_tasks // world_size
                remainder = num_tasks % world_size

                start = rank * tasks_per_rank + min(rank, remainder)
                end = start + tasks_per_rank + (1 if rank < remainder else 0)
                return tasks[start: end]
            else:
                if rank < num_tasks:
                    return tasks[rank: rank + 1]
                else:
                    return []

    def evaluate(self, eval_file, **judge_kwargs):
        eval_categorys = self.data['category'].unique().tolist()
        # custom imports to avoid environment conflict
        if any(value in eval_categorys for value in ['Color', 'Shape', 'Texture', 'Complex']):
            from .utils.t2i_compbench.blip_eval import BLIP_eval
        if any(value in eval_categorys for value in ['non_Spatial', 'Complex']):
            from .utils.t2i_compbench.clip_eval import CLIP_eval
        if any(value in eval_categorys for value in ['Spatial', 'Complex']):
            from .utils.t2i_compbench.uni_det_eval import UniDet_eval
            Image.LINEAR = Image.BILINEAR  # for detectron2
        if 'Complex' in eval_categorys:
            from .utils.t2i_compbench.complex_score import complex_score

        default_eval_methods = {
            'Color': ['BLIP'],
            'Shape': ['BLIP'],
            'Texture': ['BLIP'],
            'Spatial': ['UniDet'],
            'non_Spatial': ['CLIP'],
            'Complex': ['BLIP', 'CLIP', 'UniDet'],
        }

        default_data_index = {
            'Color': list(range(300)),
            'Shape': list(range(300, 600)),
            'Texture': list(range(600, 900)),
            'Spatial': list(range(900, 1200)),
            'non_Spatial': list(range(1200, 1500)),
            'Complex': list(range(1500, 1800)),
        }

        eval_tasks = []

        data = load(eval_file)
        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}',
                                       '_score_result.csv')
        RANK = int(os.environ.get('RANK', 0))
        WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

        for eval_cat in eval_categorys:
            eval_data = data[data['index'].isin(default_data_index[eval_cat])]
            for eval_method in default_eval_methods[eval_cat]:
                eval_tasks.append((eval_cat, eval_method, eval_data))

        rank_tasks = self.get_tasks_for_rank(eval_tasks, RANK, WORLD_SIZE)

        has_complex = False

        all_scores = {}

        for subset, eval_method, task_data in rank_tasks:
            print(f"start {subset}/{eval_method} on rank {RANK}")
            out_dir = os.path.join(
                os.path.dirname(eval_file),
                subset
            )
            img_save_dir = os.path.join(
                out_dir, 'samples'
            )
            if not os.path.exists(img_save_dir):  # create temp save images
                os.makedirs(img_save_dir)
            self.save_temp_images(task_data, img_save_dir)
            eval_func = eval(f'{eval_method}_eval')
            subset_score = eval_func(out_dir, rank=RANK, complex=(subset == 'Complex'))
            if subset == 'Complex':
                has_complex = True
            else:
                all_scores[subset] = subset_score
            print(f"finish {subset}/{eval_method} on rank {RANK}")

        if WORLD_SIZE > 1:
            dist.barrier()
        if RANK == 0:
            for subset in eval_categorys:
                out_dir = os.path.join(
                    os.path.dirname(eval_file),
                    subset
                )
                img_save_dir = os.path.join(
                    out_dir, 'samples'
                )
                self.remove_temp_images(img_save_dir)
            if has_complex:
                out_dir = os.path.join(
                    os.path.dirname(eval_file),
                    'Complex'
                )
                complex_3in1_score = complex_score(out_dir)
                all_scores["Complex"] = complex_3in1_score
            scores_pd = pd.DataFrame([all_scores])
            scores_pd.to_csv(score_file, index=False, encoding='gbk')
        return
