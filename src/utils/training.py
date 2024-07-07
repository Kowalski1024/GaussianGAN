import torch
from torch.utils.data import Dataset
from src.dataset import SyntheticDataset
from torchvision import transforms
import PIL.Image
import numpy as np
from pathlib import Path


class AverageValueMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0.0

    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n


def get_dataset(path: str | Path) -> Dataset:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = SyntheticDataset(path, transform)
    return dataset


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def fibonacci_sphere(samples=1000):
    phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))  # golden angle in radians

    indices = torch.arange(samples)
    y = 1 - (indices / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = torch.sqrt(1 - y * y)  # radius at y

    theta = phi * indices  # golden angle increment

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    points = torch.stack([x, y, z], dim=-1)

    return points


def setup_snapshot_image_grid(
    dataset: Dataset, grid_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    label_groups = {}
    grid_w, grid_h = grid_size

    for i in range(len(dataset)):
        label = dataset._load_raw_label(i)
        label = tuple(label.flatten().tolist())

        if len(label_groups) == grid_h and all(
            len(group) == grid_w for group in label_groups.values()
        ):
            break

        group = label_groups.get(label, [])
        if len(group) >= grid_w or (len(group) == 0 and len(label_groups) == grid_h):
            continue

        group.append(dataset[i][0])
        label_groups[label] = group

    images = np.stack(
        [image for group in label_groups.values() for image in group], axis=0
    )
    labels = np.stack([label for label in label_groups.keys() for _ in range(grid_w)], axis=0)

    return images, labels


def create_image_grid(
    img: np.ndarray, drange: tuple[float, float], grid_size: tuple[int, int]
) -> PIL.Image:
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    grid_w, grid_h = grid_size
    _, C, H, W = img.shape
    img = img.reshape([grid_h, grid_w, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([grid_h * H, grid_w * W, C])

    assert C == 3
    return PIL.Image.fromarray(img, "RGB")
