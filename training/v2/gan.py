import torch
from torch import nn
from .layers import Attention, EdgeBlock, AdaptivePointNorm
from .gaussian import GaussianDecoder, render
from .camera import extract_cameras, generate_cameras
from training import networks_stylegan2 as stylegan2


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
    def __init__(self, k_neighbors, z_dim, z_norm, use_attn=False, blocks=3):
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

import numpy as np

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))

    return pc / m


def load_sphere(path, points=4096):
    sphere = np.loadtxt(f"{path}/{points}.xyz")
    return pc_normalize(sphere)


class ImageGenerator(nn.Module):
    def __init__(self, attention=False, blocks=3, **kwargs):
        super(ImageGenerator, self).__init__()
        self.gaussians = GaussiansGenerator(20, 128, False, attention, blocks)
        self.decoder = GaussianDecoder()
        self.num_ws = 2048
        
        sphere = load_sphere("spheres", self.num_ws)
        sphere = torch.tensor(sphere, dtype=torch.float32).unsqueeze(0)
        
        self.register_buffer("background", torch.ones(3, dtype=torch.float32))
        self.register_buffer("sphere", sphere)

    def forward(self, ws, *args, **kwargs):
        cameras = generate_cameras(ws.size(0), ws.device)
        spheres = self.sphere.repeat(ws.size(0), 1, 1)
        gaussians = self.gaussians(spheres, ws)

        images = []
        for gaussian, camera, sphere in zip(gaussians, cameras, spheres):
            gaussian_model = self.decoder(gaussian, sphere)
            image = render(camera, gaussian_model, self.background, use_rgb=True)
            images.append(image)

        return torch.stack(images, dim=0).contiguous()


class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs = {}, **kwargs):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = 128
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = ImageGenerator(**kwargs)
        self.num_ws = 2048
        self.mapping = stylegan2.MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=128, num_ws=self.num_ws, **mapping_kwargs)


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws)
        

