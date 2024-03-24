import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
import numpy as np

from pointnet2_ops.pointnet2_utils import grouping_operation
from gaussian import GaussianModel, extract_cameras, Camera, render


neg = 0.01
neg_2 = 0.2


class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.channels = channels

        self.theta = nn.Conv1d(self.channels, self.channels // 8, 1, bias=False)
        self.phi = nn.Conv1d(self.channels, self.channels // 8, 1, bias=False)
        self.g = nn.Conv1d(self.channels, self.channels // 2, 1, bias=False)
        self.o = nn.Conv1d(self.channels // 2, self.channels, 1, bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)))

        return self.gamma * o + x


class EdgeFeature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """

    def __init__(self, k=16):
        super(EdgeFeature, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k

    def forward(self, point_cloud):
        B, dims, N = point_cloud.shape

        # batched pair-wise distance
        _, idx = self.KNN(point_cloud, point_cloud)
        idx = idx[:, 1:, :].contiguous()

        # gather
        neighbors = grouping_operation(point_cloud, idx.contiguous().int()).permute(
            0, 1, 3, 2
        )

        # centralize
        central = point_cloud.unsqueeze(3)  # [B, d, N, 1]
        central = central.repeat(1, 1, 1, self.k)  # [B, d, N, k]

        edge_feature = torch.cat([central, neighbors - central], dim=1)
        assert edge_feature.shape == (B, 2 * dims, N, self.k)

        return edge_feature, idx


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        Conv = nn.Conv1d  # EqualConv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class EdgeBlock(nn.Module):
    """Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """

    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.edge_features = EdgeFeature(k)
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout // 2, 1),
            nn.BatchNorm2d(Fout // 2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout // 2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.conv_out = nn.Conv2d(
            Fout, Fout, [1, k], [1, 1]
        )  # Fin, Fout, kernel_size, stride

    def forward(self, x):
        B, C, N = x.shape
        x, _ = self.edge_features(x)  # [B, 2Fin, N, k]
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x


class UpBlock(nn.Module):
    def __init__(self, up_ratio=4, in_channels=130):
        super(UpBlock, self).__init__()
        self.up_ratio = up_ratio
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1), nn.ReLU()
        )

        self.attention_unit = Attention(in_channels)
        self.register_buffer("grid", self.gen_grid(up_ratio))

    def forward(self, inputs):
        net = inputs  # b,128,n
        grid = self.grid.clone()
        grid = grid.unsqueeze(0).repeat(net.shape[0], 1, net.shape[2])  # b,4,2*n
        grid = grid.view([net.shape[0], -1, 2])  # b,4*n,2

        net = net.permute(0, 2, 1)  # b,n,128
        net = net.repeat(1, self.up_ratio, 1)  # b,4n,128
        net = torch.cat([net, grid], dim=2)  # b,n*4,130

        net = net.permute(0, 2, 1)  # b,130,n*4

        net = self.attention_unit(net)

        net = self.conv1(net)
        net = self.conv2(net)

        return net

    @staticmethod
    def gen_grid(up_ratio):
        import math

        sqrted = int(math.sqrt(up_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (up_ratio % i) == 0:
                num_x = i
                num_y = up_ratio // i
                break
        grid_x = torch.linspace(-0.2, 0.2, num_x)
        grid_y = torch.linspace(-0.2, 0.2, num_y)

        x, y = torch.meshgrid([grid_x, grid_y], indexing="xy")
        grid = torch.stack([x, y], dim=-1)  # 2,2,2
        grid = grid.view([-1, 2])  # 4,2
        return grid


class DownBlock(nn.Module):
    def __init__(self, up_ratio=4, in_channels=128):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=256,
                kernel_size=[up_ratio, 1],
                padding=0,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1), nn.ReLU()
        )
        self.up_ratio = up_ratio

    def forward(self, inputs):
        net = inputs

        net = net.view([inputs.shape[0], inputs.shape[1], self.up_ratio, -1])

        net = self.conv1(net)
        net = net.squeeze(2)
        net = self.conv2(net)
        return net


class UpProjection(nn.Module):
    def __init__(self, up_ratio=4):
        super(UpProjection, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1), nn.ReLU()
        )
        self.up_block1 = UpBlock(up_ratio=up_ratio, in_channels=128 + 2)
        self.up_block2 = UpBlock(up_ratio=up_ratio, in_channels=128 + 2)
        self.down_block = DownBlock(up_ratio=up_ratio, in_channels=128)

    def forward(self, input):
        # L = self.conv1(input)

        H0 = self.up_block1(input)
        # L0 = self.down_block(H0)

        # E0 = L0 - L
        # H1 = self.up_block2(E0)
        # H2 = H0 + H1
        return H0


class GaussianDecoder(nn.Module):
    feature_channels = {"scaling": 3, "rotation": 4, "opacity": 1, "shs": 3, "xyz": 3}
    use_rgb = True

    def __init__(self):
        super(GaussianDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(640, 128, 1),
            nn.SiLU(),
            nn.Linear(128, 128, 1),
            nn.SiLU(),
            nn.Linear(128, 128, 1),
            nn.SiLU(),
        )

        self.decoders = torch.nn.ModuleList()

        for key, channels in self.feature_channels.items():
            layer = nn.Linear(128, channels, 1)

            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.1))

            self.decoders.append(layer)

    def forward(self, x):
        if self.mlp is not None:
            x = self.mlp(x)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                v = torch.clamp(v, min=0, max=0.02)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                if self.use_rgb:
                    v = torch.sigmoid(v)
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                v = torch.tanh(v * 0.1) * 0.6
            ret[k] = v

        return GaussianModel(**ret)


class GaussiansGenerator(nn.Module):
    def __init__(
        self, number_points, k_neighbors, z_dim, z_norm, use_attn, use_head, off
    ):
        super(GaussiansGenerator, self).__init__()
        self.np = number_points
        self.nk = k_neighbors // 2
        self.nz = z_dim
        self.z_norm = z_norm
        self.off = off
        self.use_attn = use_attn
        self.use_head = use_head

        Conv = nn.Conv1d  # EqualConv1d
        Linear = nn.Linear  # EqualLinear

        dim = 128
        self.head = nn.Sequential(
            Conv(3 + self.nz, dim, 1),
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim, dim, 1),
            # nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_attn:
            self.attn = Attention(dim + 512)

        self.global_conv = nn.Sequential(
            Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(neg, inplace=True),
        )

        # self.tail = nn.Sequential(
        #     nn.Conv1d(512 + dim, 256, 1),
        #     nn.LeakyReLU(neg, inplace=True),
        #     nn.Conv1d(256, 64, 1),
        #     nn.LeakyReLU(neg, inplace=True),
        #     nn.Conv1d(64, 3, 1),
        #     nn.Tanh(),
        # )

        self.decoder = GaussianDecoder()

        # self.projection = UpProjection()

        if self.use_head:
            self.pc_head = nn.Sequential(
                Conv(3, dim // 2, 1),
                nn.LeakyReLU(inplace=True),
                Conv(dim // 2, dim, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.EdgeConv1 = EdgeBlock(dim, dim, self.nk)
            self.adain1 = AdaptivePointNorm(dim, dim)
            self.EdgeConv2 = EdgeBlock(dim, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)
        else:
            self.EdgeConv1 = EdgeBlock(3, 64, self.nk)
            self.adain1 = AdaptivePointNorm(64, dim)
            self.EdgeConv2 = EdgeBlock(64, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)

        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)

    def forward(self, x, z):
        B, N, _ = x.size()
        if self.z_norm:
            z = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        # x2 = self.projection(x2)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x2), dim=1)

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        feat_cat = feat_cat.permute(0, 2, 1).contiguous()
        return [self.decoder(features) for features in feat_cat]


class Generator(nn.Module):
    def __init__(self, number_points, k_neighbors, z_dim):
        super(Generator, self).__init__()
        self.gaussians = GaussiansGenerator(
            number_points, k_neighbors, z_dim, False, False, False, False
        )
        self.register_buffer("background", torch.ones(3, dtype=torch.float32))

    def forward(self, x, z, c):
        poses = c[:, :16].view(-1, 4, 4).detach()
        intrinsics = c[:, 16:25].view(-1, 3, 3).detach() * 512

        cameras = extract_cameras(poses, intrinsics)
        gaussians = self.gaussians(x, z)

        images = []
        for gaussian, camera in zip(gaussians, cameras):
            images.append(render(camera, gaussian, self.background, use_rgb=True))

        return torch.stack(images, dim=0).contiguous()
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=128):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # increased number of output channels
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),  # increased number of channels
            nn.Conv2d(64, 128, 4, 2, 1),  # increased number of output channels
            nn.BatchNorm2d(128),  # increased number of channels
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),  # increased number of channels
            nn.Conv2d(128, 256, 4, 2, 1),  # added new layer
            nn.BatchNorm2d(256),  # added new layer
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),  # added new layer
            nn.Conv2d(256, 1, 4, 1, 0),  # final layer
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)