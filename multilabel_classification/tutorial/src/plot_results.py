import matplotlib.pyplot as plt
from jax import numpy as jnp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import n_epochs


def Plot_results(train_cost: list, val_cost: list, train_acc: list, val_acc: list) -> None:
    """
    Plot training and validating results after each epoch.
    :return: None.
    """
    step = 50
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,4.5))
    axs[0].plot(train_cost, label="Train cost", color='royalblue')
    axs[0].plot(val_cost, label="Validation cost", color='tab:blue')
    axs[1].plot(train_acc, label='Train accuracy', color='orangered')
    axs[1].plot(val_acc, label='Validation accuracy', color='orange')
    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")
    axs[0].set_xticks(jnp.arange(0, n_epochs + step, step))
    axs[1].set_xticks(jnp.arange(0, n_epochs + step, step))
    axs[0].legend()
    axs[1].legend()


def Plot_Confusion_Matrix(test_estimation: list[float], y_true: jnp.array) -> ConfusionMatrixDisplay:
    """
    Creates confusion matrix and displays it.
    :return: (ConfusionMatrixDisplay) displayed confusion matrix.
    """
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=test_estimation)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot()
    return display
