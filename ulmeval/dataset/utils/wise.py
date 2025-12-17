from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
from ulmeval.smp import load  # adjust path if needed
from io import BytesIO
import base64
from PIL import Image


# Category buckets and overall aggregation
WISE_DIMENSIONS = {
    'CULTURE': ['CULTURE'],
    'TIME': ['TIME'],
    'SPACE': ['SPACE'],
    'BIOLOGY': ['BIOLOGY'],
    'PHYSICS': ['PHYSICS'],
    'CHEMISTRY': ['CHEMISTRY'],
    'overall': [],
}


def encode_image_to_base64(img_pil, max_side=320, quality=60):
    if not isinstance(img_pil, Image.Image):
        img_pil = Image.open(BytesIO(img_pil))
    img = img_pil.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


# Prompts for generation and scoring
SYSTEM_CAL_SCORE_PROMPT = (
    "You are a professional image quality auditor. "
    "Evaluate the image quality strictly according to the protocol."
)

SYSTEM_GENER_PRED_PROMPT = (
    "You are an intelligent chatbot designed for generating images based on a detailed description.\n"
    "------\n"
    "INSTRUCTIONS:\n"
    "- Read the detailed description carefully.\n"
    "- Generate an image based on the detailed description.\n"
    "- The image should be a faithful representation of the detailed description."
)

USER_GENER_PRED_PROMPT = (
    "Please generate an image based on the following description:\n"
    "{prompt}\n"
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the image."
)


def prepare_response_prompt(item: pd.Series) -> str:
    """Build the text prompt for the image generator."""
    if isinstance(item, pd.Series):
        item = item.to_dict()
    return USER_GENER_PRED_PROMPT.format(prompt=item['prompt'])


def prepare_score_prompt(item):
    """Build the scoring prompt as a joined string with short physical lines (flake8 E501-safe)."""
    # Convert pandas Series to dict if needed
    if isinstance(item, pd.Series):
        item = item.to_dict()
    if not isinstance(item['prediction'], list):
        item['prediction'] = [item['prediction']]
    prompts = []
    for image in item['prediction']:
        lines = [
            "You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE "
            "RUTHLESSNESS.",
            "Only images meeting the HIGHEST standards should receive top scores.",
            "",
            "**Input Parameters**",
            "- PROMPT: [User's original prompt]",
            "- EXPLANATION: [Further explanation of the original prompt]",
            "---",
            "",
            "## Scoring Criteria",
            "",
            "**Consistency (0-2):** How accurately and completely the image reflects the PROMPT.",
            "* **0 (Rejected):** Fails to capture key elements of the prompt, or contradicts the prompt.",
            "* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, "
            "or not accurately. Noticeable deviations from the prompt's intent.",
            "* **2 (Exemplary):** Perfectly and completely aligns with the PROMPT. Every single element and "
            "nuance of the prompt is flawlessly represented in the image. The image is an ideal, "
            "unambiguous visual realization of the given prompt.",
            "",
            "**Realism (0-2):** How realistically the image is rendered.",
            "* **0 (Rejected):** Physically implausible and clearly artificial. Breaks fundamental laws of "
            "physics or visual realism.",
            "* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements. While somewhat "
            "believable, noticeable flaws detract from realism.",
            "* **2 (Exemplary):** Achieves photorealistic quality, indistinguishable from a real photograph. "
            "Flawless adherence to physical laws, accurate material representation, and coherent spatial "
            "relationships. No visual cues betraying AI generation.",
            "",
            "**Aesthetic Quality (0-2):** The overall artistic appeal and visual quality of the image.",
            "* **0 (Rejected):** Poor aesthetic composition, visually unappealing, and lacks artistic merit.",
            "* **1 (Conditional):** Demonstrates basic visual appeal, acceptable composition, and color "
            "harmony, but lacks distinction or artistic flair.",
            "* **2 (Exemplary):** Possesses exceptional aesthetic quality, comparable to a masterpiece. "
            "Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating "
            "artistic style. Demonstrates a high degree of artistic vision and execution.",
            "",
            "---",
            "",
            "## Output Format",
            "",
            "**Do not include any other text, explanations, or labels.** Return exactly three lines:",
            "Consistency: <0|1|2>",
            "Realism: <0|1|2>",
            "Aesthetic Quality: <0|1|2>",
            "",
            "---",
            "",
            "**IMPORTANT Enforcement:**",
            "Be EXTREMELY strict. A score of '2' is rare and only for the very best images.",
            "If in doubt, downgrade.",
            "",
            "For Consistency, '2' means complete, flawless adherence to every aspect of the prompt.",
            "For Realism, '2' means virtually indistinguishable from a real photograph.",
            "For Aesthetic Quality, '2' requires exceptional artistic merit.",
            "",
            "---",
            "Here are the inputs for this evaluation:",
            f'PROMPT: "{item["prompt"]}"',
            f'EXPLANATION: "{item["explanation"]}"',
            "IMAGE (PNG, base64 data URL):",
            "data:image/png;base64, " + encode_image_to_base64(image),
            "",
            "Please strictly follow the criteria and output template."
        ]
        prompts.append("\n".join(lines))
    return prompts


def calculate_wiscore(consistency: int, realism: int, aesthetic_quality: int) -> float:
    """Weighted score in [0, 1]."""
    return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2.0


def parse_score_dict(raw: Any) -> Optional[Tuple[int, int, int]]:
    """Parse model raw output into three integers (consistency, realism, aesthetic_quality)."""
    if isinstance(raw, dict):
        def pick(d, *keys):
            return {k: d[k] for k in keys if k in d}
        d = pick(raw, 'consistency', 'realism', 'aesthetic_quality')
        if len(d) == 3:
            return int(d['consistency']), int(d['realism']), int(d['aesthetic_quality'])
        return 0, 0, 0

    s = "" if raw is None else str(raw).strip()
    mC = re.search(r'Consistency\s*:\s*([0-2])', s, re.I)
    mR = re.search(r'Realism\s*:\s*([0-2])', s, re.I)
    mA = re.search(r'(Aesthetic\s*Quality|Aesthetic)\s*:\s*([0-2])', s, re.I)
    if mC and mR and mA:
        return int(mC.group(1)), int(mR.group(1)), int(mA.group(2))

    nums = re.findall(r'(?<!\d)[0-2](?!\d)', s)
    if len(nums) == 3:
        return int(nums[0]), int(nums[1]), int(nums[2])
    return 0, 0, 0


def get_dimension_rating(mode, data_path: str) -> Dict[str, Dict[str, str]]:
    """Aggregate scores per discipline and compute overall weighted score."""
    data = load(data_path)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Loaded score file must be a pandas DataFrame.")

    bucket = {k: [] for k in WISE_DIMENSIONS}
    for i in range(len(data)):
        row = data.iloc[i]
        discipline = str(row.get('discipline', '')).strip().upper()
        scores = row.get('score')
        if not isinstance(scores, list):
            scores = [scores]
        parsed_scores = [parse_score_dict(s) for s in scores]
        parsed_scores = [p for p in parsed_scores if p is not None]
        if not parsed_scores:
            continue

        c = float(np.mean([p[0] for p in parsed_scores]))
        r = float(np.mean([p[1] for p in parsed_scores]))
        a = float(np.mean([p[2] for p in parsed_scores]))

        wiscore = calculate_wiscore(c, r, a)
        if discipline in bucket:
            bucket[discipline].append(wiscore)
        bucket['overall'].append(wiscore)

    def mean2(xs):
        xs = [x for x in xs if x is not None and x >= 0]
        if not xs:
            return 0.0
        return float(np.round(np.mean(xs), 2))

    coarse_valid = {k: mean2(v) for k, v in bucket.items()}

    if mode:
        overall = (
            0.4 * coarse_valid.get('CULTURE', 0.0)
            + 0.167 * coarse_valid.get('TIME', 0.0)
            + 0.133 * coarse_valid.get('SPACE', 0.0)
            + 0.1 * coarse_valid.get('BIOLOGY', 0.0)
            + 0.1 * coarse_valid.get('PHYSICS', 0.0)
            + 0.1 * coarse_valid.get('CHEMISTRY', 0.0)
        )
        coarse_valid['OVERALL'] = round(overall, 2)
    return dict(
        coarse_valid=coarse_valid
    )
