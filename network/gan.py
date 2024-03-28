import torch
from torch import nn
from network.layers import Attention, EdgeBlock, AdaptivePointNorm
from network.gaussian import GaussianDecoder, render
from network.camera import extract_cameras, generate_cameras


class SyntheticBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_neighbors):
        super(SyntheticBlock, self).__init__()
        self.edgeconv = EdgeBlock(in_channels, out_channels, k_neighbors)
        self.adain = AdaptivePointNorm(out_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.edgeconv(x)
        x = self.adain(x, style)
        return self.lrelu(x)


class GaussiansGenerator(nn.Module):
    def __init__(self, k_neighbors, z_dim, z_norm, use_attn=False, blocks=2):
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

        self.synthetic = nn.ModuleList()
        self.synthetic.append(SyntheticBlock(3, dim, self.nk))
        for _ in range(blocks - 1):
            self.synthetic.append(SyntheticBlock(dim, dim, self.nk))

    def forward(self, x, z):
        B, N, _ = x.size()

        if self.z_norm:
            z = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        # pc = self.sphere_encoder(x)
        x = x.transpose(2, 1).contiguous()

        for block in self.synthetic:
            x = block(x, style)

        feat_global = torch.max(x, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x), dim=1)

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
        for gaussian, camera, sphere in zip(gaussians, cameras, x):
            gaussian_model = self.decoder(gaussian, sphere)
            image = render(camera, gaussian_model, self.background, use_rgb=True)
            images.append(image)

        return torch.stack(images, dim=0).contiguous()
    

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        skip_x = self.skip(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + skip_x


class MinibatchStdLayer(nn.Module):
    def __init__(self):
        super(MinibatchStdLayer, self).__init__()

    def forward(self, x):
        std = x.std(dim=0).mean().expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], 1)


class DiscriminatorEpilogue(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=8):
        super(DiscriminatorEpilogue, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self, img_size=128, channels=[64, 128, 256, 512], dropout=0.0):
        super(Discriminator, self).__init__()
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(in_channels, out_channels, dropout=dropout)
            for in_channels, out_channels in zip([3] + channels[:-1], channels)
        ])
        self.minibatch_std = MinibatchStdLayer()
        self.epilogue = DiscriminatorEpilogue(channels[-1])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.minibatch_std(x)
        x = self.epilogue(x)
        return x
