import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from data.DAMON.dataset import DAMON
from lib.models.model import get_model
from lib.utils.eval_utils import compute_contact_metrics
from lib.utils.logger import ColorLogger


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='siglip2-giant', choices=['siglip2-giant'])
parser.add_argument('--checkpoint', default='release_checkpoint/heco_final_siglip2_giant_checkpoint.ckpt')
parser.add_argument('--test_name', default='DAMON', choices=['DAMON'])
parser.add_argument('--data_root', default='data/DAMON/data')
parser.add_argument('--output_path', default='outputs/test/DAMON')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()


torch.set_num_threads(4)
device = torch.device(args.device)


output = Path(args.output_path)
output.mkdir(parents=True, exist_ok=True)
logger = ColorLogger(str(output))


dataset = DAMON(args.data_root)
logger.info(f"# of test samples: {len(dataset)}")


loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


model = get_model(args.checkpoint, device)


raw, targets, names, indices = [], [], [], []
precision_sum, recall_sum = 0.0, 0.0
positive_samples = 0


test_iterator = tqdm(enumerate(loader), total=len(loader), leave=False)


with logging_redirect_tqdm(loggers=[logger._logger]), torch.inference_mode():
    for idx, batch in test_iterator:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(batch['image'].to(device))

        batch_logits = logits.float().cpu()
        raw.append(batch_logits)
        targets.append(batch['contact'])
        names.extend(batch['name'])
        indices.extend(batch['index'].tolist())


        batch_metrics = compute_contact_metrics(batch_logits.sigmoid().numpy(), batch['contact'].numpy())
        positive_samples += batch_metrics['positive_samples']
        precision_sum += batch_metrics['precision'] * batch_metrics['positive_samples']
        recall_sum += batch_metrics['recall'] * batch_metrics['positive_samples']

        total_cont_pre = precision_sum / positive_samples if positive_samples else 0.0
        total_cont_rec = recall_sum / positive_samples if positive_samples else 0.0
        total_cont_f1 = 2 * total_cont_pre * total_cont_rec / (total_cont_pre + total_cont_rec) if total_cont_pre + total_cont_rec else 0.0


        logger.info(f"C-Pre: {total_cont_pre:.3f} | C-Rec: {total_cont_rec:.3f} | C-F1: {total_cont_f1:.3f}")


raw = torch.cat(raw)
targets = torch.cat(targets).numpy()

if not torch.isfinite(raw).all():
    raise ValueError('Non-finite contact predictions')


metrics = compute_contact_metrics(raw.sigmoid().numpy(), targets)
metrics['excluded_rows_without_visible_head'] = dataset.excluded_rows


np.save(output / 'logits.npy', raw.numpy())
np.save(output / 'contact_gt.npy', targets)
(output / 'samples.json').write_text(json.dumps({'names': names, 'indices': indices}, indent=2) + '\n')
(output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


logger.info('Test finished!!!!')
logger.info("Final Results --- C-Pre: {:.3f} | C-Rec: {:.3f} | C-F1: {:.3f}".format(metrics['precision'], metrics['recall'], metrics['f1']))