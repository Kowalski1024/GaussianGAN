from src.datasets.dataset_base import Dataset
import numpy as np
import PIL.Image


def setup_snapshot_image_grid(
    dataset: Dataset,
    use_labels: bool = True,
    max_grid_size: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rnd = np.random.RandomState(seed)
    gh = np.clip(7680 // dataset.image_size, 4, max_grid_size)
    gw = np.clip(7680 // dataset.image_size, 4, max_grid_size)

    all_indices = list(range(len(dataset)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images, labels = zip(*[dataset[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


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
