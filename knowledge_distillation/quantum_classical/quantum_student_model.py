import pennylane as qml
import torch
from torch import nn
#import numpy as np
#import matplotlib.pyplot as plt


n_qubits = 10
n_layers = 1
params_per_gate = 3
device = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(device, interface="torch")
def qnn(inputs, weights):
    qml.AmplitudeEmbedding(inputs, range(n_qubits), pad_with=0, normalize=True)
    qml.Barrier(only_visual=True)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.probs(wires=[0, 1])


'''theta = np.random.rand(n_layers, n_qubits, params_per_gate)
qml.draw_mpl(qnode=qnn, decimals=2, style="default", expansion_strategy="device")(np.random.rand(784), theta)
plt.show()'''


class LightNN(nn.Module):
    def __init__(self, n_layers, n_qubits):
        super(LightNN, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, params_per_gate)}
        q_layer = qml.qnn.TorchLayer(qnn, weight_shapes)
        torch_layer = [q_layer]
        self.classifier = torch.nn.Sequential(*torch_layer)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


'''nn_light = LightNN(n_layers, n_qubits)
train(nn_light, train_loader, epochs=20, learning_rate=0.001, seed=999, device=dev_gpu)
test_accuracy_light = test(nn_light, test_loader, device=dev_gpu)

print(f"Test Accuracy: {test_accuracy_light:.2f}")'''

'''x = np.random.rand(784)
theta = np.random.rand(n_layers, n_qubits, 3)
qml.draw_mpl(qnode=qnn, decimals=2, style="default", expansion_strategy="device")(x, theta)
plt.show()'''