import pennylane as qml
from jax import numpy as jnp

class quantum_neural_network:
    def __init__(self, n_wires: int) -> None:
        self.n_wires = n_wires
    def QuantumMap(self, X_data: jnp.ndarray) -> None:
        """
            Quantum feature map composed by U3 gates.
            :arg:
                X_data: jax.numpy array input image.
            :return:
                quantum feature map to map classical features into quantum state.
        """
        idx = 0
        qml.Barrier(only_visual=True)
        for i in range(self.n_wires):
            qml.Rot(phi=X_data[idx + 0], theta=X_data[idx + 1], omega=X_data[idx + 2], wires=i)
            idx += 3
            qml.Rot(phi=X_data[idx + 0], theta=X_data[idx + 1], omega=X_data[idx + 2], wires=i)
            idx += 3
        for j in range(5):
            qml.Rot(phi=X_data[idx + 0], theta=X_data[idx + 1], omega=X_data[idx + 2], wires=j)
            idx += 3
        qml.RZ(phi=X_data[63], wires=5)  # Only 64 features
        qml.Barrier(only_visual=True)
    def QuantumRing_layer(self, params: jnp.ndarray) -> None:
        """
            Quantum ansatz composed by U3 gates and ring-like CNOT entangling scheme.
            :arg: self, params (jaa.numpy array of parameters).
            :return: to be connected to the feature map.
        """
        idx = 0
        for i in range(self.n_wires):
            qml.Rot(phi=params[0 + idx], theta=params[1 + idx], omega=params[2 + idx], wires=i)
            idx += 3
        qml.Barrier(only_visual=True)
        for j in range(self.n_wires - 1):
            qml.CRot(phi=params[idx + 0], theta=params[idx + 1], omega=params[idx + 2], wires=[j, j + 1])
            idx += 3
        qml.Barrier(only_visual=True)
        for k in range(self.n_wires):
            qml.Rot(phi=params[0 + idx], theta=params[1 + idx], omega=params[2 + idx], wires=k)
            idx += 3




