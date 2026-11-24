import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy import sparse
from skimage.filters import gaussian
from yacs.config import CfgNode

from lib.utils.hmr2.hmr2 import HMR2
from lib.utils.hmr2.vitdet import ViTDetPredictor


def load_hmr2_head_detector(device='cuda', base_data=None, crop_ratio=1.0):
    detector = SimpleNamespace(device=torch.device(device), crop_ratio=crop_ratio,
                               min_context_fraction=0.25, vertical_shift=0.0)
    base = Path(base_data) if base_data else Path(__file__).resolve().parents[2] / 'data/base_data'
    folder = base / 'pretrained_models/hmr2'
    cfg = CfgNode(json.loads((folder / 'config.json').read_text()))
    cfg.SMPL.MEAN_PARAMS = str(folder / 'smpl_mean_params.npz')
    state = torch.load(folder / 'hmr2.pth', map_location='cpu', weights_only=True, mmap=True)

    detector.model = HMR2(cfg, state).to(detector.device).eval()


    detector.detector = ViTDetPredictor(
        base / 'pretrained_models/human_detector/model_final_f05665.pkl', detector.device)


    detector.head_mapping = sparse.load_npz(base / 'conversions/smpl_to_flame.npz')
    detector.focal_length = float(cfg.EXTRA.FOCAL_LENGTH)
    detector.mean = np.asarray(cfg.MODEL.IMAGE_MEAN, np.float32)[:, None, None]
    detector.std = np.asarray(cfg.MODEL.IMAGE_STD, np.float32)[:, None, None]

    return detector


def prepare_hmr2_image(rgb, box, detector):
    center = (box[:2] + box[2:]) / 2
    width, height = box[2:] - box[:2]
    side = max(float(width) * 256 / 192, float(height))
    image = rgb
    factor = side / 256 / 2

    if factor > 1.1:
        image = gaussian(image, sigma=(factor - 1) / 2, channel_axis=2, preserve_range=True)

    source = np.array([center, center + [0, side / 2], center + [side / 2, 0]], np.float32)
    target = np.array([[128, 128], [128, 256], [256, 128]], np.float32)
    transform = cv2.getAffineTransform(source, target)
    patch = cv2.warpAffine(image, transform, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    patch = patch.transpose(2, 0, 1).astype(np.float32) / 255
    patch = torch.from_numpy((patch - detector.mean) / detector.std)[None].to(detector.device)

    return patch, center, side


@torch.inference_mode()
def run_hmr2_head_detector(orig_img, detector):
    rgb = np.asarray(orig_img.convert('RGB'))
    height, width = rgb.shape[:2]


    detections = detector.detector(rgb)
    valid = (detections.pred_classes == 0) & (detections.scores > 0.5)
    boxes = detections.pred_boxes.tensor[valid]

    if not len(boxes):
        raise ValueError('No person detected')

    person_box = boxes[detections.scores[valid].argmax()].cpu().numpy()
    patch, center, side = prepare_hmr2_image(rgb, person_box, detector)


    vertices, camera = detector.model(patch)

    cam = camera[0].float().cpu().numpy()
    focal = detector.focal_length / 256 * max(width, height)
    denominator = side * cam[0] + 1e-9
    translation = np.array([2 * (center[0] - width / 2) / denominator + cam[1],
                            2 * (center[1] - height / 2) / denominator + cam[2],
                            2 * focal / denominator])


    points = detector.head_mapping @ vertices[0].cpu().numpy() + translation
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)]

    if not len(points):
        raise ValueError('Head projection is outside the camera view')

    projected = focal * points[:, :2] / points[:, 2:] + [width / 2, height / 2]
    tight = np.concatenate([projected.min(axis=0), projected.max(axis=0)])

    if tight[2] <= 0 or tight[3] <= 0 or tight[0] >= width or tight[1] >= height:
        raise ValueError('Head projection is outside the image')


    head_center = (tight[:2] + tight[2:]) / 2
    head_center[1] += detector.vertical_shift * (tight[3] - tight[1])
    crop_side = min(max(float((tight[2:] - tight[:2]).max()) * detector.crop_ratio,
                        min(width, height) * detector.min_context_fraction, 2.0), min(width, height))
    left = int(np.clip(round(head_center[0] - crop_side / 2), 0, width - crop_side))
    top = int(np.clip(round(head_center[1] - crop_side / 2), 0, height - crop_side))
    crop_box = [left, top, min(width, left + round(crop_side)), min(height, top + round(crop_side))]

    return {'person_bbox': person_box.tolist(), 'head_bbox': tight.tolist(),
            'crop_bbox': crop_box, 'crop_ratio': detector.crop_ratio,
            'min_context_fraction': detector.min_context_fraction, 'vertical_shift': detector.vertical_shift}