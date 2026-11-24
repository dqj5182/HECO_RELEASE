from pathlib import Path

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall as L, instantiate
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import CascadeROIHeads, FastRCNNConvFCHead, FastRCNNOutputLayers


class ViTDetPredictor:
    def __init__(self, checkpoint, device='cuda'):
        config = model_zoo.get_config('common/models/mask_rcnn_vitdet.py').model

        for key in ['box_head', 'box_predictor', 'proposal_matcher']:
            config.roi_heads.pop(key)

        config.roi_heads.update(
            _target_=CascadeROIHeads,
            box_heads=[L(FastRCNNConvFCHead)(
                input_shape=ShapeSpec(channels=256, height=7, width=7),
                conv_dims=[256, 256, 256, 256], fc_dims=[1024], conv_norm='LN') for _ in range(3)],
            box_predictors=[L(FastRCNNOutputLayers)(
                input_shape=ShapeSpec(channels=1024), test_score_thresh=0.25,
                box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
                cls_agnostic_bbox_reg=True, num_classes='${...num_classes}')
                for w1, w2 in [(10, 5), (20, 10), (30, 15)]],
            proposal_matchers=[L(Matcher)(thresholds=[threshold], labels=[0, 1],
                                         allow_low_quality_matches=False) for threshold in [0.5, 0.6, 0.7]],
        )

        config.backbone.net.embed_dim = 1280
        config.backbone.net.depth = 32
        config.backbone.net.num_heads = 16
        config.backbone.net.drop_path_rate = 0.5
        config.backbone.net.window_block_indexes = (
            list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31)))

        self.model = instantiate(config).to(device).eval()
        DetectionCheckpointer(self.model).load(str(Path(checkpoint)))
        self.resize = ResizeShortestEdge([1024, 1024], 1024)
        self.device = torch.device(device)


    @torch.inference_mode()
    def __call__(self, rgb):
        height, width = rgb.shape[:2]
        image = self.resize.get_transform(rgb).apply_image(rgb)
        tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)),
                                 dtype=torch.float32, device=self.device)

        return self.model([{'image': tensor, 'height': height, 'width': width}])[0]['instances']