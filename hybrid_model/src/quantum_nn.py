import torch
import pennylane as qml
import torch.nn as nn


qubits = 6
layers = 2
params_per_gate = 3
dev = qml.device(name='default.qubit', wires=qubits)


@qml.qnode(device=dev, interface="torch")
def qnn_model(inputs: torch.utils.data.DataLoader, weights: torch.tensor) -> qml.probs():
    """
    Function representing the quantum circuit.
    :param inputs: (torch.utils.data.DataLoader) input data from torch dataloader;
    :param weights: (torch.tensor) trainable weights;
    :return: qml.probs() of obtaining a class after circuit measurement.
    """
    qml.AmplitudeEmbedding(features=inputs, wires=range(qubits), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(qubits))
    return qml.probs(wires=[0, 1, 2])


class QNN(nn.Module):
    """
    Quantum model class which implements quantum neural network flow.
    """
    def __init__(self, layers: int, qubits: int) -> None:
        super(QNN, self).__init__()
        weight_shapes = {"weights": (layers, qubits, params_per_gate)}
        q_layer = qml.qnn.TorchLayer(qnn_model, weight_shapes)
        torch_layer = [q_layer]
        self.qlayer = torch.nn.Sequential(*torch_layer)

    @classmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.qlayer(x)
        return x