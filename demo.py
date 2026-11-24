import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lib.models.model import get_model
from lib.utils.demo_utils import load_hmr2_head_detector, run_hmr2_head_detector
from lib.utils.preprocessing import prepare_image
from lib.utils.vis_utils import ContactRenderer


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='siglip2-giant', choices=['siglip2-giant'])
parser.add_argument('--checkpoint', default='release_checkpoint/heco_final_siglip2_giant_checkpoint.ckpt')
parser.add_argument('--input_path', default='asset/example_images')
parser.add_argument('--output_path', default='outputs')
parser.add_argument('--device', default='cuda')
parser.add_argument('--crop_ratio', type=float, default=1.0)
args = parser.parse_args()


torch.set_num_threads(4)


source = Path(args.input_path)
paths = sorted(x for x in source.iterdir() if x.suffix.lower() in {'.png', '.jpg', '.jpeg'}) if source.is_dir() else [source]

if not paths:
    raise FileNotFoundError('No input images found')

device = torch.device(args.device)


detector = load_hmr2_head_detector(device, crop_ratio=args.crop_ratio)


model = get_model(args.checkpoint, device)
contact_renderer = ContactRenderer()


output = Path(args.output_path)

for directory in ['crop_img', 'contact', 'predictions']:
    (output / directory).mkdir(parents=True, exist_ok=True)


with torch.inference_mode():
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        print(f"Processing: {path.name}")

        with Image.open(path) as image:
            rgb = image.convert('RGB')

        crop_info = run_hmr2_head_detector(rgb, detector)
        bbox = crop_info['crop_bbox']

        tensor = prepare_image(rgb, bbox)
        crop = rgb.crop(tuple(bbox))


        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            probability = model(tensor[None].to(device)).float().sigmoid()[0].cpu().numpy()

        if not np.isfinite(probability).all():
            raise ValueError('Non-finite contact prediction')


        crop.save(output / 'crop_img' / (path.stem + '.png'))

        contact_rendered = contact_renderer.render_contact(crop, probability)
        contact_rendered.save(output / 'contact' / (path.stem + '.png'))

        np.save(output / 'predictions' / (path.stem + '.npy'), probability)
        (output / 'predictions' / (path.stem + '.json')).write_text(json.dumps(crop_info, indent=2))


print('Results saved to', args.output_path)