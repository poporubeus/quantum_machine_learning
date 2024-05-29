import torch
import torch.nn as nn


cpu_device = torch.device("cpu")


class ClassicalModel(nn.Module):
    """
    A classical model class which implements classical model flow.
    """
    def __init__(self) -> None:
        super(ClassicalModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.Linear(512, 8)
        )

    @classmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


