import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Đảm bảo x có đúng kích thước
        if len(x.shape) != 4:
            if len(x.shape) == 3:
                B, C, T = x.shape
                x = x.unsqueeze(-1)  # (B, C, T, 1)
            elif len(x.shape) == 2:
                B, C = x.shape
                x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        x_shift = x + self.shift
        # Chuyển đổi tensor để phù hợp với phép nhân ma trận
        x_permuted = x_shift.permute(0, 2, 3, 1)  # (B, T, J, C)

        # Nhân với ma trận weight
        x_conv = torch.matmul(x_permuted, self.weight)  # (B, T, J, out_channels)

        # Chuyển lại định dạng
        x_conv = x_conv.permute(0, 3, 1, 2)  # (B, out_channels, T, J)

        x_conv = self.bn(x_conv)
        x_conv = self.relu(x_conv)

        return x_conv

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels//4, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)

        scale1 = self.conv1(x)
        scale3 = self.conv3(x)
        scale5 = self.conv5(x)
        scale7 = self.conv7(x)

        multi_scale = torch.cat([scale1, scale3, scale5, scale7], dim=1)
        multi_scale = self.bn(multi_scale)
        multi_scale = self.relu(multi_scale)

        return multi_scale