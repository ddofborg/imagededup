from pathlib import PurePath
import os
import pickle
from typing import Dict, Callable, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from imagededup.utils.image_utils import load_image
from imagededup.utils.general_utils import generate_files


class ImgDataset(Dataset):
    def __init__(
        self,
        image_dir: PurePath,
        basenet_preprocess: Callable[[np.array], torch.tensor],
        recursive: Optional[bool],
    ) -> None:
        self.image_dir = image_dir
        self.basenet_preprocess = basenet_preprocess
        self.recursive = recursive
        self.image_files = sorted(
            generate_files(self.image_dir, self.recursive)
        )  # ignore hidden files

    def __len__(self) -> int:
        """Number of images."""
        return len(self.image_files)

    def __getitem__(self, item) -> Dict:
        filename_encoded = self.image_files[item].with_suffix('.encoding.pickle')
        if os.path.exists(filename_encoded):
            return {'filename': self.image_files[item]}
        im_arr = load_image(self.image_files[item], target_size=None, grayscale=None)
        if im_arr is not None:
            img = self.basenet_preprocess(im_arr)
            return {'image': img, 'filename': self.image_files[item]}
        else:
            return {'image': None, 'filename': self.image_files[item]}


def _collate_fn(batch: List[Dict]) -> Tuple[torch.tensor, str, str]:
    ims, filenames, bad_images, filenames_encoded = [], [], [], []

    for b in batch:
        if not 'image' in b:
            filenames_encoded.append(b['filename'])
        else:
            im = b['image']
            if im is not None:
                ims.append(im)
                filenames.append(b['filename'])
            else:
                bad_images.append(b['filename'])
    return torch.stack(ims) if ims else None, filenames, bad_images, filenames_encoded


def img_dataloader(
    image_dir: PurePath,
    batch_size: int,
    basenet_preprocess: Callable[[np.array], torch.tensor],
    recursive: Optional[bool],
    num_workers: int
) -> DataLoader:
    img_dataset = ImgDataset(
        image_dir=image_dir, basenet_preprocess=basenet_preprocess, recursive=recursive
    )
    return DataLoader(
        dataset=img_dataset, batch_size=batch_size, collate_fn=_collate_fn, num_workers=num_workers
    )
