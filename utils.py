import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from dataset import SyntheticDataset
from torchvision import transforms
import PIL.Image
import numpy as np


def get_optimizers(
    model: nn.Module, config: dict
) -> torch.optim.Optimizer:
    optimizer_type = config.type
    params = config.params

    match optimizer_type.lower():
        case "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                **params,
            )
        case "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                **params,
            )
        case "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                **params,
            )
        case "rmsprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                **params,
            )
        case _:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def get_dataset(dataset_type: str, params: dict) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    match dataset_type.lower():
        case "synthetic":
            dataset = SyntheticDataset(**params, transform=transform)
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    return dataset


def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        sampler=RandomSampler(dataset, replacement=True, num_samples=int(10e6)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def generate_noise(batch_size, points, std=0.2, z_dim=128):
    noise = torch.normal(0, std, (batch_size, z_dim))
    return noise.unsqueeze(1).expand(-1, points, -1)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))

    return pc / m


def load_sphere(path, points=4096):
    sphere = np.loadtxt(f"{path}/{points}.xyz")
    return pc_normalize(sphere)


def save_image_grid(img, fname, drange, grid_size):
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
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)
