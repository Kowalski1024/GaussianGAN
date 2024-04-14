import torch
from network.render import ImageGenerator
from dataset import NerfDataset, CarsDataset
from torchvision import transforms as T
from loss_utils import l1_loss, ssim
import PIL.Image
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import os


DATASET_PATH = "/mnt/d/Tomasz/Pulpit/GaussianGAN/datasets/cars_train_dir"
OUTPUT_PATH = "outputs/"
BATCH_SIZE = 4
EPOCHS = 300
POINTS = 8192
IMAGE_SIZE = 128

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


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


def fibonacci_sphere(samples=1000, scale=1.0):
    phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))  # golden angle in radians

    indices = torch.arange(samples)
    y = 1 - (indices / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = torch.sqrt(1 - y * y)  # radius at y

    theta = phi * indices  # golden angle increment

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    points = torch.stack([x, y, z], dim=-1)

    return points * scale


def loss_fn(pred, gt, lambd=0.2):
    Ll1 = l1_loss(pred, gt)
    return (1.0 - lambd) * Ll1 + lambd * (1.0 - ssim(pred, gt))


def train(model, criterion, optimizer, train_loader, device, epochs):
    sphere = fibonacci_sphere(POINTS, 1.0)
    sphere = Data(pos=sphere)
    sphere.edge_index = knn_graph(sphere.pos, k=3)
    sphere = sphere.to(device)

    z = torch.randn(3, 128).to(device)

    print(f"Training epochs: {epochs}")
    for epoch in range(epochs):
        for image, camera, idx in train_loader:
            image = image.to(device)
            camera = camera.to(device)
            ws = z[idx]

            optimizer.zero_grad()
            output = model(sphere, camera, ws)
            loss = criterion(output, image)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            images = 2
            try:
                save_image_grid(
                    output[: images**2].detach().cpu().numpy(),
                    f"{OUTPUT_PATH}/output_{epoch}.png",
                    drange=[-1, 1],
                    grid_size=(images, images),
                )
                save_image_grid(
                    image[: images**2].detach().cpu().numpy(),
                    f"{OUTPUT_PATH}/gt_{epoch}.png",
                    drange=[-1, 1],
                    grid_size=(images, images),
                )
            except Exception as e:
                pass

        print(f"Epoch {epoch}, Loss: {loss.item()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CarsDataset(
        DATASET_PATH, transform=T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    background = torch.ones(3, dtype=torch.float32).to(device)

    model = ImageGenerator(background, IMAGE_SIZE).to(device)
    criterion = loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train(model, criterion, optimizer, dataloader, device, EPOCHS)


if __name__ == "__main__":
    main()
