import torch
import torch.nn as nn


class DeepNN(nn.Module):
    def __init__(self, num_classes=4):
        super(DeepNN, self).__init__()
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


'''nn_deep = DeepNN(num_classes=8)
train(nn_deep, train_loader, epochs=20, learning_rate=0.001, seed=999, device=device)
test_accuracy_deep = test(nn_deep, test_loader, device=device)

print(f"Test Accuracy: {test_accuracy_deep:.2f}")'''

