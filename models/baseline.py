from torch import flatten
import torch.nn as nn
import torch.nn.functional as F


class BaseLineModel(nn.Module):
    input_channels = 3
    num_classes = 100

    def __init__(self):
        super(BaseLineModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
