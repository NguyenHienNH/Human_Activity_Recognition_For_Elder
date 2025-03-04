import torch
import torch.nn as nn

class SENetFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features_list):
        processed_features = []
        for feature in features_list:
            if len(feature.shape) != 4:
                if len(feature.shape) == 3:
                    B, C, T = feature.shape
                    feature = feature.unsqueeze(-1)
                elif len(feature.shape) == 2:
                    B, C = feature.shape
                    feature = feature.unsqueeze(-1).unsqueeze(-1)
            processed_features.append(feature)

        combined_features = torch.cat(processed_features, dim=1)

        pooled = self.global_pool(combined_features).squeeze(-1).squeeze(-1)

        attention_weights = self.fc(pooled)

        weighted_features = combined_features * attention_weights.unsqueeze(-1).unsqueeze(-1)

        return weighted_features