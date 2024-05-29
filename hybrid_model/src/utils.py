import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from quantum_nn import QNN, layers, qubits
import pennylane as qml


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
             criterion: callable, device: torch.device) -> tuple[float, float]:
    """
    Evaluate both loss and accuracy on a specific set, in this case it shoule be the
    validation set. The loss and accuracy are computed with torch.no_grad(), i.e.
    in the evaluation mode, without computing the gradient at each epoch.
    :param model: (torch.nn.Module) model to evaluate;
    :param loader: (torch.utils.data.DataLoader) data loader;
    :param criterion: (callable) loss function;
    :param device: (torch.device) device to use;
    :return: (tuple) loss and accuracy of the model after evaluation.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_instances = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            classifications = torch.argmax(outputs, dim=1)
            correct += (classifications == labels).sum().item()
            total_instances += len(inputs)

    average_loss = total_loss / len(loader)
    accuracy = (correct / total_instances) * 100
    return average_loss, accuracy


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader, epochs: int, learning_rate: float,
          seed: int, device: callable) -> None:
    """
    Function which trains the model on training dataloader and evaluates it
    on the validation dataloader. During training, the model is set to train mode,
    it computes loss and accuracy after each epoch either on training and validation sets.
    :param model: (torch.nn.Module) the model to be trained and validated;
    :param train_loader: (torch.utils.data.DataLoader) training data loader;
    :param val_loader: (torch.utils.data.DataLoader) validation data loader;
    :param epochs: (int) number of epochs to train the model;
    :param learning_rate: (float) learning rate;
    :param seed: (int) random seed for reproducibility;
    :param device: (torch.device) device to use the model with - it can be cpu or gpu (cuda);
    :return: None.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0 # la loss che calcoliamo è una loss che si aggiorna dopo ogni epoca, non è dopo ogni batch.
        total_correct = 0
        total_instances = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            with torch.no_grad():
                classifications = torch.argmax(outputs, dim=1)
                correct_predictions = (classifications == labels).sum().item()
                total_correct += correct_predictions
                total_instances += len(inputs)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, "
            f"Validation Loss: {val_loss}, "
            f"Accuracy: {(total_correct / total_instances) * 100:.2f}%, "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Evaluate the model on the test dataloader.
    :param model: (callable) the model to evaluate;
    :param test_loader: (torch.utils.data.DataLoader) the test dataloader;
    :param device: (torch.device) the device to use for evaluation;
    :return: accuracy: (float) the accuracy on the test dataloader;
    """
    model.eval() # model settato in evaluation (non ho bisogno di calcolare il gradiente)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


class HybridQNNModel(nn.Module):
    """
    A hybrid classical quantum class which implements the operations acting on data and trainable weights.
    """

    def __init__(self) -> None:
        super(HybridQNNModel, self).__init__()
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
            nn.Linear(512, 64)  # qui obblighiamo il linear layer a passare da 512
            # features in ingresso a 64 in uscita. Queste ultime costituiranno gli inputs del quantum layer.
        )
        self.qnn = QNN(layers, qubits)

    @classmethod
    def forward(self, x: torch.Tensor) -> qml.probs():
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        outputs = self.qnn(x)  # gli outputs della rete classica vanno dentro la rete quantum.
        return outputs


