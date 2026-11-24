from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from lib.utils.preprocessing import head_box, prepare_image


class DAMON(Dataset):
    def __init__(self, root='data/DAMON/data'):
        self.root = Path(root)
        path = self.root / 'Release_Datasets/damon/hot_dca_test.npz'

        with np.load(path, allow_pickle=True) as archive:
            names = archive['imgname'].astype(str).tolist()
            parts = archive['part_seg'].astype(str).tolist()
            flame = np.load(Path(__file__).resolve().parents[1] / 'base_data/human_models/smplx/SMPL-X__FLAME_vertex_ids.npy')
            contact = (archive['contact_label_smplx'][:, flame] >= .5).astype(np.float32)


        self.rows = []
        self.excluded_rows = []

        for index, (name, part) in enumerate(zip(names, parts)):
            with Image.open(self.root / part.removeprefix('datasets/')) as image:
                box = head_box(np.asarray(image))

            if box is None:
                self.excluded_rows.append(index)
                continue


            path = self.root / name.removeprefix('datasets/')

            if not path.is_file():
                raise FileNotFoundError(path)

            self.rows.append((index, name, path, box, contact[index]))


    def __len__(self):
        return len(self.rows)


    def __getitem__(self, index):
        original, name, path, box, contact = self.rows[index]

        with Image.open(path) as image:
            tensor = prepare_image(image, box)

        return {'image': tensor, 'contact': contact, 'name': name, 'index': original}