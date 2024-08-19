import hydra
from omegaconf import DictConfig
import torch
from torchvision import transforms

from conf.optimizers import OptimizerConfig
from src.datasets.dataset_base import Dataset


def get_dataset(dataset_config: DictConfig) -> Dataset:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = hydra.utils.instantiate(dataset_config, transform=transform)
    return dataset


def adjust_optimizer(
    optimizer_cfg: OptimizerConfig,
    reg_interval: int,
) -> OptimizerConfig:
    ratio = reg_interval / (reg_interval + 1)
    optimizer_cfg.lr = optimizer_cfg.lr * ratio

    if hasattr(optimizer_cfg, "betas"):
        optimizer_cfg.betas = tuple([beta**ratio for beta in optimizer_cfg.betas])

    if hasattr(optimizer_cfg, "weight_decay"):
        optimizer_cfg.weight_decay = optimizer_cfg.weight_decay * ratio

    return optimizer_cfg


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
