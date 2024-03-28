from torch.utils.data import Dataset
from pathlib import Path
import pyspng
import numpy as np
import zipfile
import os


class SyntheticDataset(Dataset):
    def __init__(self, path, transform=None):
        super(SyntheticDataset, self).__init__()
        self.transform = transform
        self.path = path
        self.is_zip = zipfile.is_zipfile(path)

        if self.is_zip:
            self._zipfile = zipfile.ZipFile(path)
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

    def _open_file(self, fname):
        if self.is_zip:
            return self._zipfile.open(fname, "r")
        else:
            return open(os.path.join(self.path, fname), "rb")

    def _load_image(self, idx):
        fname = self._all_images[idx]

        with self._open_file(fname) as f:
            image = pyspng.load(f.read())

            if image.shape[2] == 4:  # RGBA image
                image = image[:, :, :3]  # discard alpha channel

        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC

        return image

    def _load_raw_label(self, idx):
        fname = self._all_images[idx]

        pose_path = fname.replace("rgb", "pose").replace(".png", ".txt")
        intrinsics_path = fname.replace("rgb", "intrinsics").replace("png", "txt")

        with self._open_file(pose_path) as f:
            pose = np.loadtxt(f)

        with self._open_file(intrinsics_path) as f:
            intrinsics = np.loadtxt(f) / 512.0
            intrinsics[-1] = 1.0

        return np.concatenate((pose, intrinsics), dtype=np.float32)

    def __len__(self):
        return len(self._all_images)

    def __getitem__(self, idx):
        image = self._load_image(idx)
        label = self._load_raw_label(idx)
        return self.transform(image), label
