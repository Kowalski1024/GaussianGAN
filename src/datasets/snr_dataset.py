import os
import zipfile
from pathlib import Path

import numpy as np
import pyspng
import torch

from src.datasets.dataset_base import Dataset


class SNRDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_zip = zipfile.is_zipfile(self.path)

        self._image_size = None

        if self.is_zip:
            self._zipfile = zipfile.ZipFile(self.path)
            self._all_images = [
                f.filename
                for f in self._zipfile.filelist
                if f.filename.endswith(".png")
            ]
        else:
            self._all_images = [
                str(p.relative_to(self.path)) for p in Path(self.path).rglob("*.png")
            ]

        assert len(self._all_images) > 0, f"No images found in {self._path}"

    def __len__(self) -> int:
        return len(self._all_images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(idx)
        label = self._load_raw_label(idx)

        return self.transform(image), label

    @property
    def image_size(self) -> int:
        if self._image_size is None:
            image = self._load_image(0)
            assert image.shape[0] == image.shape[1], "Image must be square"
            self._image_size = image.shape[0]
        return self._image_size

    @property
    def has_labels(self) -> bool:
        return True

    def close(self):
        if self.is_zip:
            self._zipfile.close()

    def _open_file(self, fname):
        if self.is_zip:
            return self._zipfile.open(fname, "r")
        else:
            return open(os.path.join(self.path, fname), "rb")

    def _load_image(self, idx):
        fname = self._all_images[idx]

        with self._open_file(fname) as f:
            image = pyspng.load(f.read())

        # HW => HWC
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # discard alpha channel
        if image.shape[2] == 4:
            image = image[:, :, :3]

        return image

    def _load_raw_label(self, idx):
        if not self.use_labels:
            return torch.tensor([])

        fname = self._all_images[idx]

        pose_path = fname.replace("rgb", "pose").replace(".png", ".txt")
        with self._open_file(pose_path) as f:
            pose = np.loadtxt(f)

        fov = 0.9074
        return np.concatenate((pose, np.array([fov, fov])), dtype=np.float32)
