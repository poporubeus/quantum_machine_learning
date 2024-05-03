import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert PIL image or numpy array to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the tensor with mean and standard deviation
])

# Load the MNIST training dataset and filter for classes 0 and 1
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_indices = ((train_dataset.targets == 0) | (train_dataset.targets == 1) |
                 (train_dataset.targets == 2) | (train_dataset.targets == 3))
train_dataset.data = train_dataset.data[train_indices]
train_dataset.targets = train_dataset.targets[train_indices]

# Load the MNIST testing dataset and filter for classes 0 and 1
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_indices = ((test_dataset.targets == 0) | (test_dataset.targets == 1) |
                 (test_dataset.targets == 2) | (test_dataset.targets == 3))
test_dataset.data = test_dataset.data[test_indices]
test_dataset.targets = test_dataset.targets[test_indices]

# Create data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def plot_data(dataset: torch.asarray) -> plt.figure():
    fig, axs = plt.subplots(5, 5, figsize=(10,10))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            image, label = dataset[idx]
            axs[i, j].imshow(image.squeeze(), cmap='viridis')
            axs[i, j].set_title(f"Label: {label}")
            axs[i, j].axis('off')
    plt.tight_layout()
    return fig


#plot_data(train_dataset)
#plt.show()
