# This module contains the set of unitary ansatze that will be used to benchmark the performances of Quantum Convolutional Neural Network (QCNN) in QCNN.ipynb module
import pennylane as qml
import math
import cmath
from pennylane import numpy as np
from scipy.linalg import expm, sinm, cosm
#import testPennylane


# Unitary Ansatze for Convolutional Layer
def U_1(params, wires):  # 2 params
    qml.RX(params[0], wires=wires[0])
    qml.IsingYY(params[1], wires=[wires[0], wires[1]])   


def U_2(params, wires):  # 3 params
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.IsingYY(params[2], wires=[wires[0], wires[1]])


def U_3(params, wires):  # 4 params
    qml.Rot(params[0], params[1], params[2], wires=wires[0])
    qml.Rot(params[3], params[4], params[5], wires=wires[1])
    qml.Rot(params[6], params[7], params[8], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    #qml.RY(params[2], wires=wires[0])
    #qml.RY(params[2], wires=wires[0])
    #qml.RY(params[3], wires=wires[1])
    #qml.CNOT(wires=[wires[0], wires[1]])
    #qml.RY(params[4], wires=wires[0])
    #qml.RY(params[5], wires=wires[1])
    #qml.CNOT(wires=[wires[0], wires[1]])
    
    
def U_31(params, wires):  # 4 params
    qml.Rot(params[0], params[1], params[2], wires=wires[0])
    qml.Rot(params[3], params[4], params[5], wires=wires[1])
    #qml.Rot(params[6], params[7], params[8], wires=wires[2])
    qml.CNOT(wires=[wires[0], wires[1]])
    #qml.Rot(params[6], params[7], params[8], wires=wires[0])
    #qml.Rot(params[9], params[10], params[11], wires=wires[1])
    #qml.Rot(params[6], params[7], params[8], wires=wires[2])
    #qml.CNOT(wires=[wires[0], wires[1]])
    
    #qml.CNOT(wires=[wires[1], wires[2]])
    
def U_4(params, wires):  # 5 params
    #herm1 = 1j*np.array(herm)
    #herm1 = np.array(herm1)
    #X = np.array([[0, 1], [1, 0]])
    #ID = np.array([[1, 0], [0, 1]])
    #herm = np.kron(np.kron(X, ID), X)
    #herm = (math.pi/2)*np.array(herm)
    ##Y = np.array([[0, -1j], [1j, 0]])
    #ID = np.array([[1, 0], [0, 1]])
    #herm1 = np.kron(np.kron(Y, ID), Y)
    #herm1 = (math.pi/2)*np.array(herm1)
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    #qml.RX(params[2], wires=wires[2])
    qml.IsingXX(params[2], wires=[wires[1], wires[0]])
    #qml.SWAP(wires=[wires[1], wires[0]])
    #qml.RX(params[3], wires=wires[0])
    #qml.RX(params[4], wires=wires[1])
    #qml.SWAP(wires=[wires[1], wires[0]])
    #qml.IsingXX(params[5], wires=[wires[1], wires[0]])
    #qml.RX(params[2], wires=wires[1])
    #testPennylane.RYIY(params[3], wires=[wires[0], wires[1], wires[2]])
    
    #testPennylane.RXIX(params[4], wires=[wires[0], wires[1], wires[2]])
    #testPennylane.RXIX(params[4], wires=[wires[0], wires[1], wires[2]])
    #qml.RX(params[3], wires=wires[0])
    #qml.RX(params[4], wires=wires[2])
    #testPennylane.RXIX(params[5], wires=[wires[0], wires[1], wires[2]])
    #qml.RY(params[4], wires=wires[0])
    #qml.RY(params[5], wires=wires[2])
    #qml.QubitUnitary(cosm(herm1) + 1j*sinm(herm1), wires=[wires[0], wires[1], wires[2]])
    #qml.RY(params[6], wires=wires[0])
    #qml.RY(params[7], wires=wires[2])
    #qml.QubitUnitary(cosm(herm1) + 1j*sinm(herm1), wires=[wires[0], wires[1], wires[2]])
    #qml.IsingXX(math.pi/2, wires=[wires[1], wires[0]])
    #qml.RX(params[5], wires=wires[0])
    #qml.RX(params[6], wires=wires[1])
    #qml.IsingZZ(params[7], wires=[wires[0], wires[1]])
    
    
    
def U_5(params, wires):  # 5 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])
    

# Pooling Layer

#def Pooling_ansatz1(wires):
    #qml.Hadamard(wires[0])
    #qml.CNOT(wires=[wires[0], wires[1]])
    #qml.Hadamard(wires[0])
    
#def Pooling_ansatz1(params, wires):
#    qml.Hadamard(wires[0])
#    qml.CRY(params[0], wires=[wires[0], wires[1]])
#    qml.Hadamard(wires[0])
    #qml.PauliZ(wires[0])
    #qml.Hadamard(wires[0])
    #qml.CRX(params[1], wires=[wires[0], wires[1]])
    #qml.Hadamard(wires[0])
    #qml.PauliZ(wires[0])

    
def Pooling_ansatz1(params, wires):
    qml.Hadamard(wires[0])
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.Hadamard(wires[0])
    qml.PauliZ(wires[0])
    qml.Hadamard(wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])
    qml.Hadamard(wires[0])
    qml.PauliZ(wires[0])

def Pooling_ansatz2(params, wires): #2 params
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])
    
