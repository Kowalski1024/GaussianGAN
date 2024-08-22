from pathlib import Path
from typing import Callable

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        background: tuple[int, int, int],
        use_labels: bool,
        transform: Callable | None = None,
    ):
        self.path = Path(path)
        self.use_labels = use_labels
        self.transform = transform
        self.background = background
        self.transform = transform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _load_image(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    def _load_label(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}-{self.path.name}"

    @property
    def image_size(self) -> int:
        raise NotImplementedError

    @property
    def has_labels(self) -> bool:
        raise NotImplementedError
