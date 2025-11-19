import os

import torch
from tqdm import tqdm

import json
from tqdm.auto import tqdm

import spacy

from .BLIPvqa_eval.BLIP.train_vqa_func import VQA_main


def Create_annotation_for_BLIP(image_folder, outpath, np_index=None):
    nlp = spacy.load("en_core_web_sm")

    annotations = []
    file_names = os.listdir(image_folder)
    file_names.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # sort

    cnt = 0

    # output annotation.json
    for file_name in file_names:
        image_dict = {}
        image_dict["image"] = image_folder + file_name
        image_dict["question_id"] = cnt
        f = file_name.split("_")[0]
        doc = nlp(f)

        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.text not in [
                "top",
                "the side",
                "the left",
                "the right",
            ]:  # todo remove some phrases
                noun_phrases.append(chunk.text)
        if len(noun_phrases) > np_index:
            q_tmp = noun_phrases[np_index]
            image_dict["question"] = f"{q_tmp}?"
        else:
            image_dict["question"] = ""

        image_dict["dataset"] = "color"
        cnt += 1

        annotations.append(image_dict)

    print("Number of Processed Images:", len(annotations))

    json_file = json.dumps(annotations)
    with open(f"{outpath}/vqa_test.json", "w") as f:
        f.write(json_file)


def BLIP_eval(out_dir: str, np_num: int = 8, complex=False, rank: int = 0):
    np_index = np_num  # how many noun phrases

    answer = []
    sample_num = len(os.listdir(os.path.join(out_dir, "samples")))
    reward = torch.zeros((sample_num, np_index)).to(device=f"cuda:{rank}")

    order = "_blip"  # rename file
    for i in tqdm(range(np_index)):
        print(f"start VQA{i+1}/{np_index}!")
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}", exist_ok=True)
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}/VQA/", exist_ok=True)
        Create_annotation_for_BLIP(
            f"{out_dir}/samples/",
            f"{out_dir}/annotation{i + 1}{order}",
            np_index=i,
        )
        answer_tmp = VQA_main(
            f"{out_dir}/annotation{i + 1}{order}/",
            f"{out_dir}/annotation{i + 1}{order}/VQA/",
            rank=rank
        )
        answer.append(answer_tmp)
        with open(
            f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json",
            "r",
        ) as file:
            r = json.load(file)
        with open(
            f"{out_dir}/annotation{i + 1}{order}/vqa_test.json", "r"
        ) as file:
            r_tmp = json.load(file)
        for k in range(len(r)):
            if r_tmp[k]["question"] != "":
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1
        print(f"end VQA{i+1}/{np_index}!")
    reward_final = reward[:, 0]
    for i in range(1, np_index):
        reward_final *= reward[:, i]

    # output final json
    with open(
        f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r"
    ) as file:
        r = json.load(file)
    reward_after = 0
    for k in range(len(r)):
        r[k]["answer"] = "{:.4f}".format(reward_final[k].item())
        reward_after += float(r[k]["answer"])
    os.makedirs(f"{out_dir}/annotation{order}", exist_ok=True)
    with open(f"{out_dir}/annotation{order}/vqa_result.json", "w") as file:
        json.dump(r, file)

    # calculate avg of BLIP-VQA as BLIP-VQA score
    print("BLIP-VQA score:", reward_after / len(r), "!\n")
    with open(f"{out_dir}/annotation{order}/blip_vqa_score.txt", "w") as file:
        file.write("BLIP-VQA score:" + str(reward_after / len(r)))
    return reward_after / len(r)
