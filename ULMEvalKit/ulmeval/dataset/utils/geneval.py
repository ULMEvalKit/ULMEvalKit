from ...smp import *
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import torch
import json
from clip_benchmark.metrics import zeroshot_classification as zsc
try:
    from mmdet.apis import inference_detector
except:
    logger = get_logger('MMDetModel')
    logger.critical('MMDet is not installed. Please install it if you want to evaluate on GenEval.')


CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "computer mouse",
    "tv remote", "computer keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}


# Evaluation parts


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image, objects, transform, bgcolor="#999", crop=True):
        self._image = image.convert("RGB")
        self.transform = transform
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self.crop = crop

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if self.crop:
            image = image.crop(box[:4])
        return (self.transform(image), 0)


def color_classification(clip_model, image, bboxes, classname, bgcolor="#999", crop=True, DEVICE="cuda"):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model.model, clip_model.tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                "a photo of a {{c}} object"
            ],
            DEVICE
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes, clip_model.transform, bgcolor, crop),
        batch_size=16, num_workers=4
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model.model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    def area_fn(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b, POSITION_THRESHOLD=0.1):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    #
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    #
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def evaluate(clip_model, image, objects, metadata, POSITION_THRESHOLD=0.1):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    correct = True
    reason = []
    matched_groups = []
    # Check for expected objects
    meta_classes = metadata["include_class"].split(", ")
    if isinstance(metadata["include_count"], str):
        meta_counts = metadata["include_count"].split(", ")
    else:
        meta_counts = [metadata["include_count"]]
    for i in range(len(meta_classes)):
        classname = meta_classes[i]
        counting = int(meta_counts[i])
        matched = True
        found_objects = objects.get(classname, [])[:counting]
        if len(found_objects) < counting:
            correct = matched = False
            reason.append(f"expected {classname}>={counting}, found {len(found_objects)}")
        else:
            if not pd.isna(metadata["include_color"]):
                # Color check
                color = metadata["include_color"].split(", ")[i]
                if len(color) > 0:
                    colors = color_classification(clip_model, image, found_objects, classname)
                    if colors.count(color) < counting:
                        correct = matched = False
                        reason.append(
                            f"expected {color} {classname}>={counting}, found "
                            + f"{colors.count(color)} {color}; and "
                            + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
            if not pd.isna(metadata["include_position"]) and matched:
                # Relative position check
                position = metadata["include_position"].split(", ")[i]
                if len(position) > 0:
                    expected_rel, target_group = position.split("; ")
                    target_group = int(target_group)
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj, POSITION_THRESHOLD)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    # Check for non-expected objects
    if not pd.isna(metadata["exclude_class"]):
        meta_classes = metadata["exclude_class"].split(", ")
        if isinstance(metadata["exclude_count"], str):
            meta_counts = metadata["exclude_count"].split(", ")
        else:
            meta_counts = [metadata["exclude_count"]]
        for i in range(len(meta_classes)):
            classname = meta_classes[i]
            counting = int(meta_counts[i])
            if len(objects.get(classname, [])) >= counting:
                correct = False
                reason.append(f"expected {classname}<{counting}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def GenEval_auxeval_score(det_model, clip_model, line):
    # THRESHOLD = float(args.options.get('threshold', 0.3))
    # COUNTING_THRESHOLD = float(args.options.get('counting_threshold', 0.9))
    # MAX_OBJECTS = int(args.options.get('max_objects', 16))
    # NMS_THRESHOLD = float(args.options.get('max_overlap', 1.0))
    # POSITION_THRESHOLD = float(args.options.get('position_threshold', 0.1))
    # How to set these parameters?
    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1
    result = inference_detector(det_model.model, np.array(line["prediction"]))
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    image = ImageOps.exif_transpose(line["prediction"])
    detected = {}
    # Determine bounding boxes to keep
    confidence_threshold = THRESHOLD if line["tag"] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(CLASS_NAMES):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]  # Threshold
        ordering = ordering[:MAX_OBJECTS].tolist()  # Limit number of detected objects per class
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]
    # Evaluate
    is_correct, reason = evaluate(clip_model, image, detected, line, POSITION_THRESHOLD)
    return {
        'index': line['index'],
        'tag': line['tag'],
        'prompt': line['question'],
        'correct': is_correct,
        'reason': reason,
        'details': json.dumps({
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        })
    }


def GenEval_acc(result_file):
    df = load(result_file)

    # Measure overall success
    res = defaultdict(list)
    # res['image_acc'].append(f"{df['correct'].mean():.2%}")
    # res['prompt_acc'].append(f"{df.groupby('metadata')['correct'].any().mean():.2%}")

    task_scores = []
    for tag, task_df in df.groupby('tag', sort=False):
        task_scores.append(task_df['correct'].mean())
        res['split'].append(tag)
        res['Overall'].append(
            f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})"
        )
    res['split'].append('all')
    res['Overall'].append(f"{np.mean(task_scores):.5f}")

    return pd.DataFrame(res)
