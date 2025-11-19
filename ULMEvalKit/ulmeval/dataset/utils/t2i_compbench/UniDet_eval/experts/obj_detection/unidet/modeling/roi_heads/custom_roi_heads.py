from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from .custom_fast_rcnn import CustomFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret["box_predictor"]
        ret["box_predictor"] = CustomFastRCNNOutputLayers(
            cfg, ret["box_head"].output_shape
        )
        return ret


@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CascadeROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret["box_predictors"]
        cascade_bbox_reg_weights = (
            cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        )
        box_predictors = []
        for box_head, bbox_reg_weights in zip(
            ret["box_heads"], cascade_bbox_reg_weights
        ):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(
                        weights=bbox_reg_weights
                    ),
                )
            )
        ret["box_predictors"] = box_predictors
        return ret
