import torch
import torch.nn as nn
import torchvision.models as models


class CNNTemporalAttention(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, feature_dim=None, freeze_backbone=True, use_lstm=False, lstm_hidden=256, num_classes=2):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError('backbone must be resnet18 or efficientnet_b0')

        if feature_dim is None:
            feature_dim = in_features

        self.feature_dim = feature_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=False)
            self.attention = nn.Sequential(
                nn.Linear(lstm_hidden, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)
            )
            classifier_in = lstm_hidden
        else:
            self.attention = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)
            )
            classifier_in = in_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: B, T, C, H, W
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        feats = self.backbone(x)
        feats = feats.view(b, t, -1)  # B, T, D

        if self.use_lstm:
            feats, _ = self.lstm(feats)  # B, T, H

        att = self.attention(feats)  # B, T, 1
        weights = torch.softmax(att, dim=1)

        agg = (feats * weights).sum(dim=1)  # B, D
        out = self.classifier(agg)
        return out


def get_model(**kwargs):
    return CNNTemporalAttention(**kwargs)
