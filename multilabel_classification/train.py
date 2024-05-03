from config import *
from sklearn.utils import gen_batches
from digits import X_train, y_train, X_test, y_test, X_val, y_val
from utils import *
from plot_results import Plot_results, PLot_Confusion_Matrix
import matplotlib.pyplot as plt


def training(seed: int, model: callable, selected_shape: int, n_epochs: int, batch_size: int) -> list:
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
        print(f"Epoch: {epoch}, ---Train loss: ", cost, "---Train acc: ", train_acc,
              "| ---Val loss: ", val_cost, "---Val acc: ", val_acc)
    optimal_params = params  # Optimal parameters after training saved here, ready to be used for predictions.
    return [train_cost_epochs, train_acc_epochs, val_cost_epochs, val_acc_epochs, optimal_params]


if "__main__" == __name__:
    quantum_model = SelectModel(qcnn)
    train_cost, train_acc, val_cost, val_acc, optimal_params = training(seed=seed,
                                                                        model=quantum_model,
                                                                        selected_shape=params_size,
                                                                        n_epochs=n_epochs,
                                                                        batch_size=batch_size)
    test_estimation, test_acc = TestAcc(X=X_test,
                                        y=y_test,
                                        optimal_params=optimal_params,
                                        quantum_model=quantum_model)
    print("Test acc. teacher model:", test_acc)
    Plot_results(train_cost, val_cost, train_acc, val_acc)
    PLot_Confusion_Matrix(test_estimation)
    plt.show()