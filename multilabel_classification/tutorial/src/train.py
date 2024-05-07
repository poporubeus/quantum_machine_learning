from sklearn.utils import gen_batches
from src.utils import *
import numpy as np


def training(seed: int, model: callable, selected_shape: int, n_epochs: int, batch_size: int,
             X_train: jnp.array, y_train: jnp.array, X_val: jnp.array, y_val: jnp.array) -> list:
    """
    Functions which trains the QCNN model on digits dataset with parameters chosen by the user.
    :param seed: (int) Seed for reproducibility;
    :param model: (callable) The QCNN model to train;
    :param selected_shape: (int) Number of parameters to pass within the model;
    :param n_epochs: (int) Number of epochs to train the model with;
    :param batch_size: (int) Batch size.
    :return: (list) List of values:
    - train_cost_epochs (list) List of training cost values;
    - train_acc_epochs (list) List of training accuracy values;
    - val_cost_epochs (list) List of validation cost values;
    - val_acc_epochs (list) List of validation accuracy values;
    - optimal_params (list) List of optimal parameters for the QCNN assumed to be the last ones.
    """
    updated_params = np.zeros(shape=[200, 43])
    updated_val_acc = np.zeros(shape=[200, 1])
    key = jax.random.PRNGKey(seed)
    initial_params = jax.random.normal(key, shape=(selected_shape,))
    key = jax.random.split(key)[0]
    params = jnp.copy(initial_params)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(initial_params)
    train_cost_epochs, train_acc_epochs, val_cost_epochs, val_acc_epochs = [], [], [], []

    for epoch in range(1, n_epochs + 1):
        idxs_dataset = jax.random.choice(key, jnp.array(list(range(X_train.shape[0]))), shape=(X_train.shape[0],),
                                         replace=False)
        key = jax.random.split(key)[0]
        for i in gen_batches(X_train.shape[0],
                             batch_size):
            idxs = idxs_dataset[i]
            params, opt_state, cost = optimizer_update(opt_state, params, X_train[idxs, :], y_train[idxs], model)

        cost = optax_bce(X_train, y_train, params, model)
        val_cost = optax_bce(X_val, y_val, params, model)
        train_acc = accuracy(X_train, y_train, params, model)
        val_acc = accuracy(X_val, y_val, params, model)
        train_cost_epochs.append(cost)
        val_cost_epochs.append(val_cost)
        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)
        updated_params[epoch-1, :] = params
        updated_val_acc[epoch-1, :] = val_acc
        print(f"Epoch: {epoch}, ---Train loss: ", cost, "---Train acc: ", train_acc,
              "| ---Val loss: ", val_cost, "---Val acc: ", val_acc)
    last_params = params
    optimal_params_index = tracking_params(updated_val_acc)
    optimal_params = updated_params[optimal_params_index]
    optimal_params = np.squeeze(np.reshape(optimal_params_index, newshape=(optimal_params.shape[1], optimal_params.shape[0])),0)
    return [train_cost_epochs, train_acc_epochs, val_cost_epochs, val_acc_epochs, last_params, optimal_params]

