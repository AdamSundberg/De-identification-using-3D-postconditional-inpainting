import torch
import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16 * 16, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.7),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = self.fc(x)
        return x
