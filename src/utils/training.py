import hydra
from omegaconf import DictConfig
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torchvision import transforms

from conf.optimizers import OptimizerConfig
from src.datasets.dataset_base import Dataset


def get_dataset(dataset_config: DictConfig) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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


def generate_sphere(points: int, k: int) -> Data:
    sphere = fibonacci_sphere(points)
    edge_index = knn_graph(sphere, k=k, batch=None, loop=False)
    return Data(pos=sphere, edge_index=edge_index)


def generate_noise(batch_size: int, noise_channels: int, std: float = 1.0) -> torch.Tensor:
    return torch.normal(mean=0.0, std=std, size=(batch_size, noise_channels))


def r1_penalty(
    real_logits: torch.Tensor,
    real_images: torch.Tensor,
    r1_gamma: float,
) -> torch.Tensor:
    if r1_gamma == 0:
        return torch.tensor(0.0, device=real_images.device)

    grad_real = torch.autograd.grad(
        real_logits.sum(), real_images, create_graph=True, only_inputs=True
    )[0]
    grad_penalty = grad_real.square().sum(dim=(1, 2, 3))
    grad_penalty = grad_penalty * (r1_gamma / 2)
    return grad_penalty


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
