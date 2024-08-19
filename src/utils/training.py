import torch
from src.datasets.dataset_base import Dataset
from torchvision import transforms
from omegaconf import DictConfig
import hydra


def get_dataset(dataset_config: DictConfig) -> Dataset:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = hydra.utils.instantiate(dataset_config, transform=transform)
    return dataset


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def fibonacci_sphere(samples: int) -> torch.Tensor:
    phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))

    indices = torch.arange(samples)
    y = 1 - (indices / float(samples - 1)) * 2
    radius = torch.sqrt(1 - y * y)

    theta = phi * indices

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    points = torch.stack([x, y, z], dim=-1)

    return points
