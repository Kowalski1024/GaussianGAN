from torch.utils.data import Dataset
import numpy as np
import os
import json
import math
from PIL import Image


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
        return 2*math.atan(pixels/(2*focal))

    def load_image(self, idx):
        fname = self._transforms["frames"][idx]["file_path"] + ".png"
        image = Image.open(os.path.join(self.path, fname))
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

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
