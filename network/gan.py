import torch
from torch import nn
from network.layers import Attention, EdgeBlock, AdaptivePointNorm
from network.gaussian import GaussianDecoder, render
from network.camera import extract_cameras, generate_cameras
import rff


class GaussiansGenerator(nn.Module):
    def __init__(self, k_neighbors, z_dim, z_norm, use_attn):
        super(GaussiansGenerator, self).__init__()
        self.nk = k_neighbors // 2
        self.nz = z_dim
        self.z_norm = z_norm
        self.use_attn = use_attn

        dim = 128
        if self.use_attn:
            self.attn = Attention(dim + 512)

        # self.sphere_encoder = rff.layers.PositionalEncoding(sigma=1.0, m=16)
        self.head = nn.Sequential(
            nn.Conv1d(3 + self.nz, dim, 1),
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(inplace=True),
        )

        self.global_conv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.EdgeConv1 = EdgeBlock(3, dim, self.nk)
        self.adain1 = AdaptivePointNorm(dim, dim)
        self.EdgeConv2 = EdgeBlock(dim, dim, self.nk)
        self.adain2 = AdaptivePointNorm(dim, dim)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, z):
        B, N, _ = x.size()

        if self.z_norm:
            z = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        # pc = self.sphere_encoder(x)
        pc = x.transpose(2, 1).contiguous()

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu(x2)
        x2 = self.adain2(x2, style)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x2), dim=1)

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        feat_cat = feat_cat.permute(0, 2, 1).contiguous()
        return feat_cat


class ImageGenerator(nn.Module):
    def __init__(self, k_neighbors, z_dim):
        super(ImageGenerator, self).__init__()
        self.gaussians = GaussiansGenerator(k_neighbors, z_dim, False, False)
        self.decoder = GaussianDecoder()

        self.register_buffer("background", torch.ones(3, dtype=torch.float32))

    def forward(self, x, z, camera=None):
        if camera is None:
            cameras = generate_cameras(x.size(0), x.device)
        else:
            poses = camera[:, :16].view(-1, 4, 4).detach()
            intrinsics = camera[:, 16:25].view(-1, 3, 3).detach() * 512
            cameras = extract_cameras(poses, intrinsics)

        gaussians = self.gaussians(x, z)

        images = []
        for gaussian, camera in zip(gaussians, cameras):
            gaussian_model = self.decoder(gaussian)
            image = render(camera, gaussian_model, self.background, use_rgb=True)
            images.append(image)

        return torch.stack(images, dim=0).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=128, channels=[64, 128, 256]):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.main = self._make_layers(channels)

    def _make_layers(self, channels):
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(out_channels),
                nn.BatchNorm2d(out_channels) if out_channels != channels[-1] else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 0))  # final layer
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
