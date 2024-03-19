import pennylane as qml
import unitary
import embedding

# Quantum Circuits for Convolutional layers
def conv_layer1(U, params):
    #U(params, wires=[0, 8])
    #for i in range(0, 10, 3):
        #U(params, wires=[i, i + 1, i + 2])
    #for i in range(1, 10, 3):
        #U(params, wires=[i, i + 1, i + 2])
    #for i in range(2, 10, 3):
        #U(params, wires=[i, i + 1, i + 2])
    for i in range(0, 9, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 9, 2):
        U(params, wires=[i, i + 1])
    #U(params, wires=[0, 11])
def conv_layer2(U, params):
    #U(params, wires=[0, 8])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    #U(params, wires=[8, 10])
    U(params, wires=[2, 4])
    U(params, wires=[6, 8])
    #U(params, wires=[0, 10])
    #U(params, wires=[7, 9])
    #U(params, wires=[0, 2, 4])
    #U(params, wires=[6, 8, 10])
    #U(params, wires=[2, 4, 6])
    #U(params, wires=[4, 6, 8])
    #U(params, wires=[0, 2])
    #U(params, wires=[4, 6])
    #U(params, wires=[8, 10])
    #U(params, wires=[2, 4])
    #U(params, wires=[6, 8])
def conv_layer3(U, params):
    U(params, wires=[0,4])
    U(params, wires=[4,8])
    #U(params, wires=[0,8])
    #U(params, wires=[0, 4, 8])
def conv_layer4(U, params):
    U(params, wires=[0,8])


# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 9, 2):
        V(params, wires=[i+1, i])
def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])
    #V(params, wires=[10,8])
def pooling_layer3(V, params):
    V(params, wires=[4,0])
    #V(params, wires=[8,0])
def pooling_layer4(V, params):
    V(params, wires=[8,0])
    
 # Quantum circuit for fully-connected part
#def fully_connected(params):
#    qml.RY(params[0], wires=0)
#    qml.RY(params[1], wires=2)
#    qml.RY(params[2], wires=4)
#    qml.RY(params[3], wires=6)
#    qml.CNOT(wires=[0, 2])
#    qml.CNOT(wires=[2, 4])
#    qml.CNOT(wires=[4, 6])
#    qml.RY(params[4], wires=0)
#    qml.RY(params[5], wires=2)
#    qml.RY(params[6], wires=4)
#    qml.RY(params[7], wires=6)
#    qml.CNOT(wires=[0, 2])
#    qml.CNOT(wires=[2, 4])
#    qml.CNOT(wires=[4, 6])

def QCNN_structure(U, params, U_params):
#    
#    # For Reflection and rotation invariant MNIST
#    
    param1 = params[0:U_params[0]]
    param2 = params[U_params[0]: (U_params[0]+U_params[1])]
    param3 = params[U_params[0]+U_params[1]: (U_params[0]+2*U_params[1])]
    param4 = params[U_params[0]+2*U_params[1]: (U_params[0]+3*U_params[1])]
    param5 = params[(U_params[0]+3*U_params[1]):(U_params[0]+3*U_params[1]+2)]
    param6 = params[(U_params[0]+3*U_params[1]+2):(U_params[0]+3*U_params[1]+4)]
    param7 = params[(U_params[0]+3*U_params[1]+4):(U_params[0]+3*U_params[1]+6)]
    param8 = params[(U_params[0]+3*U_params[1]+6):(U_params[0]+3*U_params[1]+8)]
#    

#    # Pooling Ansatz1 is used by default
    conv_layer1(U[0], param1)
    pooling_layer1(unitary.Pooling_ansatz1, param5)
    conv_layer2(U[0], param2)
    pooling_layer2(unitary.Pooling_ansatz1, param6)
    conv_layer3(U[0], param3)
    pooling_layer3(unitary.Pooling_ansatz2, param7)
    conv_layer4(U[0], param4)
    pooling_layer4(unitary.Pooling_ansatz2, param8)
    #fully_connected(params9)



dev = qml.device('default.qubit', wires = 10)
@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='mse'):
    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)
    QCNN_structure([unitary.U_4, unitary.U_4], params, U_params)
    
    # Quantum Convolutional Neural Network
    #if U == 'U_1':
        #QCNN_structure(unitary.U_1, params, U_params)
    #elif U == 'U_2':
        #QCNN_structure(unitary.U_2, params, U_params)
    #elif U == 'U_3':
        #QCNN_structure(unitary.U_3, params, U_params)
    #elif U == 'U_4':
        #QCNN_structure(unitary.U_4, params, U_params)
    #elif U == 'U_5':
        #QCNN_structure(unitary.U_5, params, U_params)
    #else:
        #print("Invalid Unitary Ansatze")
        #return False

    if cost_fn == 'mse':
        #result = [qml.expval(qml.PauliZ(j)) for j in [0]]
        result = qml.expval(qml.PauliZ(0))
        #result = [qml.expval(qml.PauliZ(j)) for j in [0, 4, 8]]
        #if qml.expval > result[1]:
            #prediction = 0
        #else:
            #prediction = 1
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=0)
        #prediction = result
        #result = qml.probs(wires=4)
    return result
