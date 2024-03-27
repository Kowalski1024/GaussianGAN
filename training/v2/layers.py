import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN

try:
    from pointnet2_ops.pointnet2_utils import grouping_operation
except ImportError:
    grouping_operation = None    


class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.channels = channels

        self.theta = nn.Conv1d(self.channels, self.channels // 8, 1, bias=False)
        self.phi = nn.Conv1d(self.channels, self.channels // 8, 1, bias=False)
        self.g = nn.Conv1d(self.channels, self.channels // 2, 1, bias=False)
        self.o = nn.Conv1d(self.channels // 2, self.channels, 1, bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
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

    def __init__(self, k=16, use_pointnet=True):
        super(EdgeFeature, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k
        self.use_pointnet = use_pointnet and grouping_operation is not None

    def forward(self, point_cloud):
        B, dims, N = point_cloud.shape

        # batched pair-wise distance
        xt = point_cloud.permute(0, 2, 1)
        dist = torch.cdist(xt, xt, p=2)  # [B, N, N]

        # get k NN id
        _, idx_o = torch.topk(dist, self.k+1, dim=2, largest=False)
        idx = idx_o[: ,: ,1:] # [B, N, k]
        idx = idx.contiguous().view(B, -1)

        # gather
        neighbors = []
        for b in range(B):
            tmp = torch.index_select(
                point_cloud[b], 1, idx[b]
            )  # [d, N*k] <- [d, N], 0, [N*k]
            tmp = tmp.view(dims, N, -1)
            neighbors.append(tmp)

        neighbors = torch.stack(neighbors)  # [B, d, N, k]

        # centralize
        central = point_cloud.unsqueeze(3)  # [B, d, N, 1]
        central = central.repeat(1, 1, 1, self.k)  # [B, d, N, k]

        edge_feature = torch.cat([central, neighbors - central], dim=1)
        assert edge_feature.shape == (B, 2 * dims, N, self.k)

        return edge_feature, idx


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
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
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(Fout // 2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(inplace=True),
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
        L = self.conv1(input)

        H0 = self.up_block1(L)
        L0 = self.down_block(H0)

        E0 = L0 - L
        H1 = self.up_block2(E0)
        H2 = H0 + H1
        return H2
