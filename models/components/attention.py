import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_channels = max(1, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        identity = x

        if len(x.shape) != 4:
            if len(x.shape) == 2:
                B, C = x.shape
                x = x.view(B, C, 1, 1)
            elif len(x.shape) == 3:
                B, C, L = x.shape
                x = x.view(B, C, L, 1)

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        x_cat = torch.cat([x_h, x_w], dim=2)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = self.act(x_cat)

        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w).sigmoid()

        result = identity * x_h * x_w

        if result.shape != identity.shape:
            result = result.view(identity.shape)

        return result

class AdvancedCrossStreamAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, streams):
        processed_streams = []
        for stream in streams:
            if len(stream.shape) != 4:
                if len(stream.shape) == 3:
                    B, C, T = stream.shape
                    stream = stream.unsqueeze(-1)
                elif len(stream.shape) == 2:
                    B, C = stream.shape
                    stream = stream.unsqueeze(-1).unsqueeze(-1)
            processed_streams.append(stream)

        streams = processed_streams
        B, C, T, J = streams[0].size()

        fused_features = []
        for i in range(len(streams)):
            query = self.query(streams[i]).view(B, -1, T*J)
            key = self.key(streams[(i+1)%len(streams)]).view(B, -1, T*J)
            value = self.value(streams[(i+1)%len(streams)]).view(B, C, T*J)

            energy = torch.bmm(query.permute(0,2,1), key)
            attention = F.softmax(energy, dim=-1)

            cross_feat = torch.bmm(value, attention.permute(0,2,1))
            cross_feat = cross_feat.view(B, C, T, J)

            fused_features.append(cross_feat * self.fusion_weights[i])

        return torch.mean(torch.stack(fused_features), dim=0)