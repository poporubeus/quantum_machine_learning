import pennylane as qml
import matplotlib.pyplot as plt
from jax import numpy as jnp
from plot_results import Plot_architecture


class QCNNArchitecture:
    """
    QCNN class which creates the corresponding quantum circuit.
    """
    def __init__(self, device: qml.device, wires: list) -> None:
        """
        Init function which creates the class' instance.
        :param device: (qml.device) Pennylane's device;
        :param wires: (list) List of wires to use;
        :return None.
        """
        self.device = device
        self.wires = wires

    @staticmethod
    def UGate(params: jnp.array, wires: list) -> None:
        """
        UGate function which select the rotational gates to use within the layers.
        :param params: (jnp.array) Parameters of the layer;
        :param wires: (list) List of wires to use;
        :return: None
        """
        qml.Rot(params[0], params[1], params[2], wires=wires)

    @staticmethod
    def EntanglingGate(params: jnp.array, wires1: int, wires2: int) -> None:
        """
        EntanglingGate function which creates the entanglement between wires.
        :param params: (jnp.array) Parameters of the gate within the entangling gate;
        :param wires1: (int) Index of the controlled wire;
        :param wires2: (int) Index of the target wire.
        :return: None.
        """
        qml.CNOT(wires=[wires1, wires2])
        qml.Rot(params[0], params[1], params[2], wires=wires2)
        qml.CNOT(wires=[wires1, wires2])

    @staticmethod
    def QPoolFilter_1(params: jnp.array, wires_to_measure: int, wires_to_apply: int) -> None:
        """
        QPoolFilter_1 function which acts as the first Pooling layer.
        :param params: (jnp.array) Parameters of the layer;
        :param wires_to_measure: (int) Index of the wire to measure;
        :param wires_to_apply: (int) Index of the wire to apply the conditioned gate.
        :return: None.
        """
        m_outcome = qml.measure(wires_to_measure)
        qml.cond(m_outcome, qml.Rot)(*params, wires=wires_to_apply)

    def QCNN(self, params: jnp.array) -> None:
        """
        QCNN function which collects all the layers and combined them together.
        :param params: (jnp.array) Parameters of the whole layers;
        :return: None.
        """

        # QConvNet_1
        self.UGate(params[:3], wires=self.wires[0])
        self.UGate(params[:3], wires=self.wires[1])
        self.UGate(params[3:6], wires=self.wires[2])
        self.UGate(params[3:6], wires=self.wires[3])
        self.UGate(params[6:9], wires=self.wires[4])
        self.UGate(params[6:9], wires=self.wires[5])

        # Entangling scheme
        self.EntanglingGate(params[9:12], wires1=self.wires[0], wires2=self.wires[1])
        self.EntanglingGate(params[9:12], wires1=self.wires[2], wires2=self.wires[3])
        self.EntanglingGate(params[9:12], wires1=self.wires[4], wires2=self.wires[5])

        # QConvNet_2
        self.UGate(params[:3], wires=self.wires[0])
        self.UGate(params[:3], wires=self.wires[1])
        self.UGate(params[3:6], wires=self.wires[2])
        self.UGate(params[3:6], wires=self.wires[3])
        self.UGate(params[6:9], wires=self.wires[4])
        self.UGate(params[6:9], wires=self.wires[5])
        qml.Barrier(only_visual=True)

        # QPooling_1
        self.QPoolFilter_1(params[12:15], wires_to_measure=self.wires[0], wires_to_apply=self.wires[1])
        self.QPoolFilter_1(params[12:15], wires_to_measure=self.wires[2], wires_to_apply=self.wires[3])
        self.QPoolFilter_1(params[12:15], wires_to_measure=self.wires[4], wires_to_apply=self.wires[5])
        qml.Barrier(only_visual=True)

        # QConvNet_3
        self.UGate(params[15:18], wires=self.wires[1])
        self.UGate(params[15:18], wires=self.wires[3])
        self.UGate(params[15:18], wires=self.wires[5])

        # Entangling scheme
        self.EntanglingGate(params[18:21], wires1=self.wires[1], wires2=self.wires[3])
        self.EntanglingGate(params[18:21], wires1=self.wires[3], wires2=self.wires[5])

        # QConvNet_4
        self.UGate(params[15:18], wires=self.wires[1])
        self.UGate(params[15:18], wires=self.wires[3])
        self.UGate(params[15:18], wires=self.wires[5])
        qml.Barrier(only_visual=True)

        # QPooling_2
        self.QPoolFilter_1(params[18:21], wires_to_measure=self.wires[1], wires_to_apply=self.wires[3])
        qml.ArbitraryUnitary(params[21:36], wires=[self.wires[3], self.wires[5]])


if __name__ == "__main__":
    Plot_architecture()
    plt.show()

