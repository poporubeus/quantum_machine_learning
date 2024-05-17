from probabilistic_QCNN_DATASET import new_x_train, new_x_test, Y_train, Y_test
from probabilistic_QCNN_model import qcnn_model
import jax
import jax.numpy as jnp
import optax
from sklearn.metrics import accuracy_score
from sklearn.utils import gen_batches
import warnings
warnings.filterwarnings("ignore")


jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


X_train, X_test = new_x_train, new_x_test
y_train, y_test = Y_train, Y_test
X_train = jnp.asarray(X_train)
X_test = jnp.asarray(X_test)
Y_train = jnp.asarray(Y_train)
Y_test = jnp.asarray(Y_test)


@jax.jit
def optax_bce(x: jnp.array, y: jnp.array, theta: jnp.array) -> jnp.ndarray:
    """
    Computes the cross-entropy loss between the x inputs predicted by the model and
    the actual labels belonging to the dataset. The cross-entropy is implemented
    by using an optax built-in function, which applies a softmax on the data predicted.
    :param x: (jnp.array) The input data;
    :param y: (jnp.array) The actual labels of the input data;
    :param theta: (jnp.array) The parameters of the model;
    :param quantum_model: (callable) The function which generates the quantum circuit.
    :return: loss (jnp.array) The cross-entropy loss mean between the x inputs predicted by the
    model and/or the actual labels belonging to the dataset
    """
    labels = jnp.array(y)
    pred = jnp.array(quantum_model(x, theta))
    #one_hot = jax.nn.one_hot(labels, pred.shape[1])
    one_hot = pred
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=one_hot, labels=labels))
    #loss = jnp.mean(optax.softmax_cross_entropy(logits=one_hot, labels=labels))
    return loss


@jax.jit
def optimizer_update(opt_state: optax, params: jnp.array, x: jnp.array, y: jnp.array) -> tuple:
    """
    Updates the optimizer and calculates the gradient of the loss w.r.t. the parameters.
    :param opt_state: (optax) The optimizer state;
    :param params: (jnp.array) The parameters of the model;
    :param x: (jnp.array) The input data;
    :param y: (jnp.array) The actual labels of the input data;
    :param quantum_model: (callable) The function which generates the quantum circuit.
    :return: (tuple):
    - params (jnp.array) The updated parameters;
    - opt_state (optax) The optimizer state;
    - loss (jnp.array) The cross-entropy loss mean values.
    """
    optimizer = optax.adam(learning_rate=0.01)
    loss_value, grads = jax.value_and_grad(lambda theta: optax_bce(x, y, theta))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


@jax.jit
def accuracy(X: jnp.array, y: jnp.array, params: jnp.array) -> jnp.ndarray:
    """
    Computes the accuracy of the model after each epoch.
    :param X: (jnp.array) The input data;
    :param y: (jnp.array) The actual labels of the input data;
    :param params: (jnp.array) The parameters of the model;
    :param quantum_model: (callable) The function which generates the quantum circuit.
    :return: mean_acc: (jnp.array) The mean accuracy.
    """
    labels = jnp.array(y)
    pred = jnp.array(quantum_model(X, params))
    accuracy = jnp.sum(jnp.argmax(jnp.array(pred), axis=1) == labels)
    mean_acc = accuracy/len(labels)
    return mean_acc


def TestAcc(X: jnp.array, y: jnp.array, optimal_params: jnp.array) -> tuple:
    """
    It computes the accuracy on the test set. It could be used also for training and validating
    sets.
    :param X: (jnp.array) The test dataset;
    :param y: (jnp.array) The test labels;
    :param optimal_params: (jnp.array) The optimal parameters of the QCNN;
    :param quantum_model: (callable) The quantum function of the circuit.
    :return: (tuple):
    - estimation (jnp.array) The estimated labels;
    - score (float) The score after training.
    """
    labels = jnp.array(y)
    predictions = quantum_model(X, optimal_params)
    estimation = jnp.argmax(predictions, axis=1)
    score = accuracy_score(y_true=labels, y_pred=estimation)
    return estimation, score


def training(seed: int, selected_shape: int, n_epochs: int, batch_size: int) -> list:
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
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(initial_params)
    train_cost_epochs, train_acc_epochs, = [], []

    for epoch in range(1, n_epochs + 1):
        idxs_dataset = jax.random.choice(key, jnp.array(list(range(X_train.shape[0]))), shape=(X_train.shape[0],),
                                         replace=False)
        key = jax.random.split(key)[0]
        for i in gen_batches(X_train.shape[0],
                             batch_size):
            idxs = idxs_dataset[i]
            params, opt_state, cost = optimizer_update(opt_state, params, X_train[idxs, :], y_train[idxs])
        cost = optax_bce(X_train, y_train, params)
        train_acc = accuracy(X_train, y_train, params)
        train_cost_epochs.append(cost)
        train_acc_epochs.append(train_acc)
        print(f"Epoch: {epoch}, ---Train loss: ", cost, "---Train acc: ", train_acc)
    optimal_params = params
    return [train_cost_epochs, train_acc_epochs, optimal_params]


quantum_model_batched = jax.vmap(qcnn_model, (0, None,))
quantum_model = jax.jit(quantum_model_batched)

loss_fn, acc_fn, optimal_params = training(seed=8888, selected_shape=15, n_epochs=100, batch_size=128)