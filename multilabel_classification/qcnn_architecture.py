import pennylane as qml
from digits import X_train
import numpy as np
import matplotlib.pyplot as plt


class QCNNArchitecture:
    def __init__(self, device: qml.device, wires: list) -> None:
        self.device = device
        self.wires = wires

    @staticmethod
    def UGate(params, wires):
        qml.Rot(params[0], params[1], params[2], wires=wires)

    @staticmethod
    def EntanglingGate(params, wires1, wires2):
        qml.CNOT(wires=[wires1, wires2])
        qml.Rot(params[0], params[1], params[2], wires=wires2)
        qml.CNOT(wires=[wires1, wires2])

    @staticmethod
    def QPoolFilter_1(params, wires_to_measure, wires_to_apply):
        m_outcome = qml.measure(wires_to_measure)
        qml.cond(m_outcome, qml.Rot)(*params, wires=wires_to_apply)

    def QCNN(self, params):
        # QConvNet_1
        self.UGate(params[:3], wires=self.wires[0])
        self.UGate(params[:3], wires=self.wires[1])
        self.UGate(params[3:6], wires=self.wires[2])
        self.UGate(params[3:6], wires=self.wires[3])
        self.UGate(params[6:9], wires=self.wires[4])
        self.UGate(params[6:9], wires=self.wires[5])
        # Entangling scheme
        self.EntanglingGate(params[9:12], wires1=self.wires[0], wires2=self.wires[1])
        self.EntanglingGate(params[12:15], wires1=self.wires[2], wires2=self.wires[3])
        self.EntanglingGate(params[15:18], wires1=self.wires[4], wires2=self.wires[5])
        # QConvNet_2
        self.UGate(params[18:21], wires=self.wires[0])
        self.UGate(params[18:21], wires=self.wires[1])
        self.UGate(params[21:24], wires=self.wires[2])
        self.UGate(params[21:24], wires=self.wires[3])
        self.UGate(params[24:27], wires=self.wires[4])
        self.UGate(params[24:27], wires=self.wires[5])
        qml.Barrier(only_visual=True)
        # QPooling_1
        self.QPoolFilter_1(params[27:30], wires_to_measure=self.wires[0], wires_to_apply=self.wires[1])
        self.QPoolFilter_1(params[27:30], wires_to_measure=self.wires[2], wires_to_apply=self.wires[3])
        self.QPoolFilter_1(params[27:30], wires_to_measure=self.wires[4], wires_to_apply=self.wires[5])
        qml.Barrier(only_visual=True)
        # QConvNet_3
        self.UGate(params[30:33], wires=self.wires[1])
        self.UGate(params[30:33], wires=self.wires[3])
        self.UGate(params[30:33], wires=self.wires[5])
        # Entangling scheme
        self.EntanglingGate(params[33:36], wires1=self.wires[1], wires2=self.wires[3])
        self.EntanglingGate(params[36:39], wires1=self.wires[3], wires2=self.wires[5])
        # QConvNet_4
        self.UGate(params[39:42], wires=self.wires[1])
        self.UGate(params[39:42], wires=self.wires[3])
        self.UGate(params[39:42], wires=self.wires[5])
        qml.Barrier(only_visual=True)
        # QPooling_2
        self.QPoolFilter_1(params[42:45], wires_to_measure=self.wires[1], wires_to_apply=self.wires[3])
        qml.ArbitraryUnitary(params[45:60], wires=[self.wires[3], self.wires[5]])





















