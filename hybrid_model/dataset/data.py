import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets

mean = 33.3184
std = 78.5675
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean/255, std=std/255)  # Normalize: I saw in a forum https://discuss.pytorch.org/t/mnist-normalization-and-torchvisions-normalize/113622
])

dataset = datasets.MNIST(root = './data', train=True, transform=transform, download=True)
data_indeces = ((dataset.targets == 0) | (dataset.targets == 1) |
                 (dataset.targets == 2) | (dataset.targets == 3)|
                (dataset.targets == 4) | (dataset.targets == 5) |
                (dataset.targets == 6) | (dataset.targets == 7)
                )
dataset.data = dataset.data[data_indeces]
dataset.targets = dataset.targets[data_indeces]


train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.65, 0.35])

# Load the MNIST testing dataset and filter for classes 0, 1, 2 and 3
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_indices = ((test_dataset.targets == 0) | (test_dataset.targets == 1) |
                 (test_dataset.targets == 2) | (test_dataset.targets == 3) |
                (test_dataset.targets == 4) | (test_dataset.targets == 5 ) |
                (test_dataset.targets == 6) | (test_dataset.targets == 7))

test_dataset.data = test_dataset.data[test_indices]
test_dataset.targets = test_dataset.targets[test_indices]

# Create data loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def plot_data(dataset: torch.asarray) -> plt.figure():
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            image, label = dataset[idx]
            axs[i, j].imshow(image.squeeze(), cmap='viridis')
            axs[i, j].set_title(f"Label: {label}")
            axs[i, j].axis('off')
    plt.tight_layout()
    return fig
