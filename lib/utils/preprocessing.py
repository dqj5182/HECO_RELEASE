import numpy as np
import torch
from PIL import Image


def prepare_image(image, box=None):
    image = image.convert('RGB')

    if box is not None:
        image = image.crop(tuple(int(x) for x in box))

    image = image.resize((384, 384), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(array.transpose(2, 0, 1).copy()).sub_(.5).div_(.5)


def head_box(mask, scale=1.0):
    if mask.ndim == 3:
        mask = mask[..., 0]

    ys, xs = np.where(mask == 16)

    if not len(xs):
        return None

    height, width = mask.shape
    tight = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
    center = (tight[:2] + tight[2:]) / 2
    side = min(float(np.max(tight[2:] - tight[:2]) * scale), float(width), float(height))

    x0 = np.clip(center[0] - side / 2, 0, width - side)
    y0 = np.clip(center[1] - side / 2, 0, height - side)

    return np.array([np.floor(x0), np.floor(y0), np.ceil(x0 + side), np.ceil(y0 + side)], dtype=np.int64)