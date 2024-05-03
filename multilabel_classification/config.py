import pennylane as qml


params_size = 37
lr = 0.001
seed = 2222
wires = list(range(6))
dev = qml.device("default.qubit", wires=6)
n_epochs = 200
batch_size = 32