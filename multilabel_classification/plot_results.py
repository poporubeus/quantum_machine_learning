import matplotlib.pyplot as plt
from jax import numpy as jnp
from train import train_acc, train_cost, val_acc, val_cost, test_estimation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from digits import y_test, X_train
from qcnn_architecture import QCNNArchitecture
from config import dev, wires, n_epochs
import pennylane as qml
import numpy as np


def Plot_results() -> plt.figure():
    """
    Plot training and validating results after each epoch.
    :return: plt.figure()
    """
    step = 10
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4.5))
    axs[0].plot(train_cost, label="Train cost teacher", color='royalblue')
    axs[0].plot(val_cost, label="Validation cost teacher", color='tab:blue')
    axs[1].plot(train_acc, label='Train accuracy teacher', color='orangered')
    axs[1].plot(val_acc, label='Validation accuracy teacher', color='orange')
    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")
    axs[0].set_xticks(jnp.arange(0, n_epochs + step, step))
    axs[1].set_xticks(jnp.arange(0, n_epochs + step, step))
    axs[0].legend()
    axs[1].legend()
    return fig


def PLot_Confusion_Matrix() -> ConfusionMatrixDisplay:
    """
    Creates confusion matrix and displays it.
    :return: (ConfusionMatrixDisplay) displayed confusion matrix.
    """
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=test_estimation)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot()
    return display


def Plot_architecture() -> qml.draw_mpl:
    """
    Plot the architecture.
    :return: qml.draw_mpl.
    """
    @qml.qnode(dev, interface="jax")
    def qnn_circuit(x: jnp.array, w: jnp.array) -> qml.probs:
        """
        QNN architecture which call the qnode and simulates the circuit.
        :param x: (jnp.array) input data;
        :param w: (jnp.array) weights;
        :return: (list) probabilities of each measured classes.
        """
        circuit = QCNNArchitecture(device=dev, wires=wires)
        qml.AmplitudeEmbedding(features=x, wires=range(len(wires)), normalize=True, pad_with=0.)
        circuit.QCNN(w)
        probs = qml.probs(wires=[3, 5])
        return probs
    weights = np.random.rand(37)
    return qml.draw_mpl(qnode=qnn_circuit)(X_train, weights)

