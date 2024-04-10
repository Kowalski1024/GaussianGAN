import torch
import torch.nn as nn
from dataset import SyntheticDataset, NerfDataset
from torchvision import transforms
from network.gaussian import GaussianDecoder, render
from network.camera import extract_cameras, extract_cameras_2
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import KNNGraph, Compose
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import (
    MessagePassing,
    global_max_pool,
    PointGNNConv,
    global_add_pool,
    global_mean_pool,
    EdgeConv,
)
from loss_utils import l1_loss, ssim
from utils import save_image_grid, fibonacci_sphere
import rff
from torch.cuda.amp import autocast, GradScaler

batch_size = 16
# car_state = torch.load("car1.pth")
# xyz_raw = car_state["xyz"]
# random sample only 32k points
# idx = torch.randperm(xyz_raw.size(0))[:32000]
# xyz_raw = xyz_raw[:8000, :]
xyz_raw = fibonacci_sphere(100000) * 0.3
# normalize
# mean = xyz_raw.mean(dim=0)
# xyz_raw = xyz_raw - mean
# m = torch.max(torch.sqrt(torch.sum(xyz_raw**2, dim=1)))
# xyz_raw = xyz_raw / m
xyz = Data(pos=xyz_raw)
xyz = KNNGraph(k=3)(xyz)
normalize = transforms.Compose([transforms.ToTensor()])
dataset = NerfDataset("datasets/lego", transform=normalize)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)


class EdgeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.conv = EdgeConv(self.mlp, aggr="sum")

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class Block(nn.Module):
    def __init__(self, din_in, dim_out):
        super().__init__()

        self.mlp_h = nn.Sequential(
            nn.Linear(din_in, din_in),
            nn.ReLU(inplace=True),
            nn.Linear(din_in, 3),
            nn.Tanh(),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(din_in + 3, dim_out),
            nn.ReLU(inplace=True),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, din_in),
            nn.ReLU(inplace=True),
        )

        self.network = PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g, aggr="sum")

    def forward(self, x, pos, edge_index):
        return self.network(x, pos, edge_index)


class PointGenerator(nn.Module):
    def __init__(self):
        super(PointGenerator, self).__init__()

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=64
        )


        self.global_conv = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 3),
            nn.Tanh(),
        )

        self.conv1 = Block(128, 128)
        self.conv2 = Block(128, 128)

    def forward(self, pos, edge_index, batch):
        x = self.encoder(pos)
        x = self.conv1(x, pos, edge_index)
        x = self.conv2(x, pos, edge_index)

        h = global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=1)

        return self.tail(x), x


class GaussiansGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.point_encoder = PointGenerator()

        # self.encoder = rff.layers.GaussianEncoding(
        #     sigma=10.0, input_size=3, encoded_size=64
        # )

        self.block1 = Block(128, 128)
        self.block2 = Block(128, 128)
        # self.block3 = Block(128, 128)
        # self.block4 = Block(128, 256)

        self.global_conv = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, pos, edge_index, batch):
        pos, x = self.point_encoder(pos, edge_index, batch)
        x = self.global_conv(x)
        # x = self.encoder(pos)

        x = self.block1(x, pos, edge_index)
        x = self.block2(x, pos, edge_index)
        # x = self.block3(x, pos, edge_index)
        # x = self.block4(x, pos, edge_index)
        # x = self.block3(x, pos, edge_index)

        # h = global_mean_pool(x, batch)
        # h = self.global_conv(h)
        # h = h.repeat(x.size(0), 1)
        # x = torch.cat([x, h], dim=1)

        return x, pos


class ImageGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussians = GaussiansGenerator()
        self.decoder = GaussianDecoder(128)

        self.register_buffer("background", torch.ones(3, dtype=torch.float32))

    def forward(self, x: Data, camera=None):
        poses = camera[:, :16].view(-1, 4, 4).detach()
        fovx = camera[:, 16].detach()
        fovy = camera[:, 17].detach()
        cameras = extract_cameras_2(poses, fovx, fovy)

        gaussians, pos = self.gaussians(x.pos, x.edge_index, x.batch)

        gaussian_model = self.decoder(gaussians, pos)

        images = []
        for camera in cameras:
            image = render(camera, gaussian_model, self.background, use_rgb=True)
            images.append(image)

        out = torch.stack(images, dim=0).contiguous()
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xyz = xyz.to(device)
xyz_raw = xyz_raw.to(device)
generator = ImageGenerator().to(device)
generator.train()
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
mse = nn.MSELoss()

# num of parameters
print(sum(p.numel() for p in generator.parameters() if p.requires_grad))
# load model
state = torch.load("generator_3.pth")
# print(state.pop("xyz_raw"))
generator.load_state_dict(state)



scaler = GradScaler()

for epoch in range(300):
    print(f"Epoch {epoch}")
    for image, camera in dataloader:
        image = image.to(device)
        camera = camera.to(device)

        # optimizer.zero_grad()
        with torch.no_grad():
            output = generator(xyz, camera)
        # Ll1 = l1_loss(output, image)
        # loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(output, image))

        # scaler.scale(loss).backward()
        # optimizer.step()

    images = 2
    try:
        save_image_grid(
            output[: images**2].detach().cpu().numpy(),
            f"output.png",
            drange=[0, 1],
            grid_size=(images, images),
        )
        # save_image_grid(
        #     image[: images**2].detach().cpu().numpy(),
        #     f"image.png",
        #     drange=[0, 1],
        #     grid_size=(images, images),
        # )
    except ValueError:
        pass
    print(f"Epoch {epoch}, Loss: {loss.item()}")

    if epoch % 10 == 0:
        torch.save(generator.state_dict(), "generator_3.pth")
