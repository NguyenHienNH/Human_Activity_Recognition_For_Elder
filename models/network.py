import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.attention import CoordinateAttention, AdvancedCrossStreamAttention
from models.components.convolution import ShiftGCN, MultiScaleFeatureExtractor
from models.components.transformer import SwinTransformerLayer, StochasticDepth
from models.components.fusion import SENetFusionModule

class EnhancedMHGTN(nn.Module):
    def __init__(self, num_joints=25, num_classes=55, num_streams=3):
        super().__init__()

        self.d_model = 64
        self.num_streams = num_streams

        # Tính toán số kênh sau fusion
        self.fusion_channels = self.d_model * (num_streams + 1)  # +1 cho cross-stream features

        # Multi-Scale Feature Extractors
        self.multi_scale_extractors = nn.ModuleList([
            MultiScaleFeatureExtractor(2, self.d_model) for _ in range(num_streams)
        ])

        # Stream Encoders
        self.stream_encoders = nn.ModuleList([
            nn.Sequential(
                ShiftGCN(self.d_model, self.d_model),
                SwinTransformerLayer(self.d_model, 8),
                CoordinateAttention(self.d_model),
                StochasticDepth(0.1)
            ) for _ in range(num_streams)
        ])

        # Cross-Stream Attention
        self.cross_stream_attention = AdvancedCrossStreamAttention(self.d_model)

        # SENet Fusion
        self.senet_fusion = SENetFusionModule(self.fusion_channels)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, *streams):
        # Đảm bảo mỗi stream có đúng kích thước
        processed_streams = []
        for stream in streams:
            # Chuyển đổi (batch, frames, joints, dims) sang (batch, dims, frames, joints)
            stream = stream.permute(0, 3, 1, 2)
            processed_streams.append(stream)

        # Multi-scale feature extraction
        multi_scale_features = [
            extractor(stream) for extractor, stream in
            zip(self.multi_scale_extractors, processed_streams)
        ]

        # Stream encoding
        encoded_features = [
            encoder(multi_scale_feat) for encoder, multi_scale_feat
            in zip(self.stream_encoders, multi_scale_features)
        ]

        # Cross-stream interaction
        cross_stream_features = self.cross_stream_attention(encoded_features)

        # SENet fusion
        fused_features = self.senet_fusion(
            [cross_stream_features] + encoded_features
        )

        # Global pooling
        global_features = F.adaptive_avg_pool2d(fused_features, 1).squeeze(-1).squeeze(-1)

        # Classification
        output = self.classifier(global_features)

        return output