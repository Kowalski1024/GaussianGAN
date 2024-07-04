import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from src.dataset import SyntheticDataset
from torchvision import transforms
import PIL.Image
import numpy as np


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def fibonacci_sphere(samples=1000):
    phi = torch.pi * (3. - torch.sqrt(torch.tensor(5.)))  # golden angle in radians

    indices = torch.arange(samples)
    y = 1 - (indices / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = torch.sqrt(1 - y*y)  # radius at y

    theta = phi * indices  # golden angle increment

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    points = torch.stack([x, y, z], dim=-1)

    return points


def save_image_grid(img, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        return PIL.Image.fromarray(img[:, :, 0], "L")
    if C == 3:
        return PIL.Image.fromarray(img, "RGB")