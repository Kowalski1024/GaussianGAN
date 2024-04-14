from torch.utils.data import Dataset
import numpy as np
import os
import json
import math
from PIL import Image
from pathlib import Path
import pyspng


class CarsDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.transform = transform
        self.path = path
        self._all_images = []
        self._all_folders = set()

        for file in Path(self.path).rglob("*.png"):
            self._all_images.append(str(file))
            self._all_folders.add(file.parent.parent.name)

        self.folder_mapping = {
            folder: idx for idx, folder in enumerate(self._all_folders)
        }

        assert len(self._all_images) > 0, f"No images found in {self._path}"

    def _open_file(self, fname):
        return open(os.path.join(self.path, fname), "rb")

    def _get_folder(self, idx):
        fname = self._all_images[idx]
        return fname.split("/")[0]

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
        fname = self._all_images[idx]
        folder_name = Path(fname).parent.parent.name
        folder_idx = self.folder_mapping[folder_name]
        return self.transform(image), label, folder_idx


class NerfDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.transform = transform
        self.path = path

        self._trainform_file = os.path.join(self.path, "transforms_train.json")
        self._transforms = json.load(open(self._trainform_file))
        self._fovx = self._transforms["camera_angle_x"]
        self._image_size = 800

    @staticmethod
    def fov2focal(fov, pixels):
        return pixels / (2 * math.tan(fov / 2))

    @staticmethod
    def focal2fov(focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))

    def load_image(self, idx):
        fname = self._transforms["frames"][idx]["file_path"] + ".png"
        image = Image.open(os.path.join(self.path, fname))
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
            1 - norm_data[:, :, 3:4]
        )
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        return image

    def load_raw_label(self, idx):
        pose = self._transforms["frames"][idx]["transform_matrix"]
        FoVx = self._fovx
        FoVy = self.focal2fov(self.fov2focal(FoVx, self._image_size), self._image_size)

        pose = np.array(pose, dtype=np.float32).flatten()
        return np.concatenate((pose, np.array([FoVx, FoVy])), dtype=np.float32)

    def __len__(self):
        return len(self._transforms["frames"])

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_raw_label(idx)
        return self.transform(image), label
