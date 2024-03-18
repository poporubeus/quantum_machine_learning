# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
import numpy as np


#def basis_transformed_encoding(X):
    #s1 = np.matrix([[0,1],[1,0]])
    #ID = np.identity(2)
    #A = np.kron(np.kron(np.kron(ID, s1), np.kron(ID, s1)), np.kron(np.kron(s1, ID), np.kron(s1, ID))) # for 8 qubits #
    #A = np.kron(np.kron(np.kron(np.kron(ID, s1), np.kron(ID, s1)), np.kron(np.kron(ID, ID), np.kron(s1, ID))), np.kron(s1, ID)) # for 10 qubits #
    #Y = np.dot(A, X)
    #Y = np.array(Y)
    #Z = np.squeeze(Y)
    #return AmplitudeEmbedding(Z, wires=range(8), normalize=True)
    
#def basis_transformed_encoding_rgb(X):
#    s1 = np.matrix([[0,1],[1,0]])
#    ID = np.identity(2)
#    A = np.kron(np.kron(np.kron(np.kron(ID, s1), np.kron(ID, s1)), np.kron(np.kron(ID, ID), np.kron(s1, ID))), np.kron(np.kron(s1, ID), np.kron(ID, s1)))
#    Y = np.dot(A, X)
#    Y = np.array(Y)
#    Z = np.squeeze(Y)
#    return AmplitudeEmbedding(Z, wires=range(12), normalize=True)


def data_embedding(X, embedding_type='Amplitude'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(10), normalize=True)
        #basis_transformed_encoding(X)
        
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(12), rotation='Y')
    elif embedding_type == 'Angle-compact':
        AngleEmbedding(X[:8], wires=range(8), rotation='X')
        AngleEmbedding(X[8:16], wires=range(8), rotation='Y')

