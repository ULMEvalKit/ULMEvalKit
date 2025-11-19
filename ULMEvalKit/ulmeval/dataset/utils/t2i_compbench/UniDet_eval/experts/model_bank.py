# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torchvision.transforms as transforms


def load_expert_model(task = None, ckpt = None, rank: int = None):
    if task == "obj_detection":
        # UniDet is wrapped in detection2,
        # the model takes input in the format of: {"image": image (BGR), "height": height, "width": width}
        import argparse
        from detectron2.engine.defaults import DefaultPredictor
        from .obj_detection.utils import setup_cfg
        import inspect
        import os
        from yacs.config import CfgNode as CN

        file_path = inspect.getfile(inspect.currentframe())
        cur_dir = os.path.dirname(os.path.abspath(file_path))
        cfgnode = CN()
        cfgnode.confidence_threshold = 0.5
        if ckpt == "RS200":
            cfgnode.config_file = os.path.join(
                cur_dir,
                "obj_detection/configs/Unified_learned_OCIM_RS200_6x+2x.yaml",
            )
            cfgnode.opts = [
                "MODEL.WEIGHTS",
                os.path.join(
                    cur_dir,
                    "expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth",
                ),
            ]
        elif ckpt == "R50":
            cfgnode.config_file = os.path.join(
                cur_dir,
                "obj_detection/configs/Unified_learned_OCIM_R50_6x+2x.yaml",
            )
            cfgnode.opts = [
                "MODEL.WEIGHTS",
                os.path.join(
                    cur_dir,
                    "expert_weights/Unified_learned_OCIM_R50_6x+2x.pth",
                ),
            ]
        else:
            raise ValueError("Invalid checkpoint")

        cfg = setup_cfg(cfgnode)
        cfg.defrost()
        cfg.MULTI_DATASET.UNIFIED_LABEL_FILE = os.path.join(
            cur_dir, "..", cfg.MULTI_DATASET.UNIFIED_LABEL_FILE
        )
        cfg.MODEL.DEVICE = rank
        cfg.freeze()
        model = DefaultPredictor(cfg).model
        transform = transforms.Compose(
            [transforms.Resize(size=479, max_size=480)]
        )
    else:
        print("Task not supported")
        model = None
        transform = None

    model.eval()
    return model, transform
