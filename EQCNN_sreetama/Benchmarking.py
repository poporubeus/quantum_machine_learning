import data
import Training
import QCNN_circuit
import numpy as np
import pennylane as qml
import autograd.numpy as anp
from pennylane import numpy as np

import random
import time
from datetime import datetime


def accuracy_test(predictions, labels, cost_fn, binary = True):
    if cost_fn == 'mse':
        if binary == True:
            acc = 0
            for l, p in zip(labels, predictions):
                prediction = np.amax(p)
                if np.abs(l - prediction) < 1:
                    acc = acc + 1
            return acc / len(labels)

        else:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc = acc + 1
            return acc / len(labels)

    elif cost_fn == 'cross_entropy':
        acc = 0
        for l,p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)


def Encoding_to_Embedding(Encoding):
    # Amplitude Embedding / Angle Embedding
    if Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    elif Encoding == 'autoencoder8':
        Embedding = 'Angle'
    return Embedding
    
    
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        prediction = np.amax(p)
        loss = loss + (l - prediction) ** 2
    loss = loss / len(labels)
    return loss
    

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss
    

def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN':
        predictions = np.array([QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X])
    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss


def Benchmarking(dataset, classes, Unitaries, U_num_params, Encoding, circuit, cost_fn, processNumber, binary=True):

    np.random.seed(processNumber+2)
    f = open(f'results/cifar10/cifar10_eqv_{str(processNumber)}.txt', 'w')
    Embedding = Encoding_to_Embedding(Encoding)
    U = Unitaries
    U_params = U_num_params
    X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes, feature_reduction=Encoding, binary=binary)
    
    learning_rate = 0.01
    batch_size = 128
    total_params = U_params[0] + 3*U_params[1] + 8
    #total_params = U_params[0] + 2 + 2*4
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    #opt = qml.AdamOptimizer(stepsize=learning_rate, beta1=0.5, beta2=0.9, eps=1e-08)
    
    steps = 500
    accuracy_sum = 0
    loss_history = []
    print("Proc " + str(processNumber) + " Loss History for " + circuit + " circuits, " + str(Unitaries) + " " + Encoding + " with " + str(cost_fn) + " steps " + str(steps))
    params = np.random.randn(total_params, requires_grad=True)
    #print(params)
    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        if it % 10 == 0:
            #train_predictions = np.array([QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn = cost_fn) for x in X_train])
            #train_accuracy = accuracy_test(train_predictions, Y_train, cost_fn, binary)
            #print("Train Accuracy after step " + str(it-1) + ": " + str(accuracy))
            test_predictions = np.array([QCNN_circuit.QCNN(x, params, U, U_params, Embedding, cost_fn = cost_fn) for x in X_test])
            #print("Proc " + str(processNumber) + " " +str(it) + "	" + str(test_predictions))
            test_accuracy = accuracy_test(test_predictions, Y_test, cost_fn, binary)
            print("Proc " + str(processNumber) + " " +str(it) + "	" + str(test_accuracy))
            print("\n")
            #f.write(str(it) + "	" + str(test_accuracy))
            #f.write("\n")
        
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, Embedding, circuit, cost_fn),
                                                     params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("Proc " + str(processNumber) + " iteration: " + str(it+1) + " cost: ", str(cost_new))
            
        #if it%200 == 0:
            #print("Iteration " + str(it) + "params: " + str(params))
        
    predictions = np.array([QCNN_circuit.QCNN(x, params, Unitaries, U_num_params, Embedding, cost_fn) for x in X_test])        
    accuracy = accuracy_test(predictions, Y_test, cost_fn, binary)
    print("Proc " + str(processNumber) + " " + str(it+1) + "	" + str(accuracy))
    print("\n")
    #f.write(str(it+1) + "	" + str(accuracy))
    #f.write("\n")
             
    #print("Proc " + str(processNumber) + " Avg Accuracy for " + str(Unitaries) + " " + Encoding + "steps" + str(steps) + " :" + str(avg_accuracy))
    #f.write("Loss History for " + circuit + " circuits, " + str(Unitaries) + " " + Encoding + " with " + cost_fn + "steps" + str(steps))
       
       
    f.close()


