import jax
from jax import numpy as jnp
import optax
from sklearn.metrics import accuracy_score
from config import lr
import numpy as np


def tracking_params(val: np.ndarray) -> int:
    """
    Function which extracts the index of the optimized parameters corresponding to the max value of the
    model's validation accuracy.
    :param val: (np.array) Validation accuracy of the model;
    :return: (int) Index of the optimal parameter.
    """
    index_max_val_acc = np.argmax(val, axis=0)
    return index_max_val_acc


def SelectModel(circuit_function: callable) -> jax.jit:
    """
    Select the model to use in the experiment. Note that any model is allowed to be used
    within this experiment, but it needs to have the same parameters as the circuit function.
    :param circuit_function: (callable) The function which generates the circuit.
    :return: quantum_model (jax.jit) The quantum circuit mapped and executed.
    """
    qnn_model = jax.vmap(
        circuit_function,
        (0, None)
    )
    quantum_model = jax.jit(qnn_model)
    return quantum_model


def optax_bce(X: jnp.array, y: jnp.array, theta: jnp.array, quantum_model: callable) -> jnp.ndarray:
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
    pred = jnp.array(quantum_model(X, theta))
    one_hot = jax.nn.one_hot(labels, pred.shape[1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=pred, labels=one_hot))
    return loss


def optimizer_update(opt_state: optax, params: jnp.array, x: jnp.array, y: jnp.array, quantum_model: callable) -> tuple:
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
    optimizer = optax.adam(learning_rate=lr)
    loss_value, grads = jax.value_and_grad(lambda theta: optax_bce(x, y, theta, quantum_model))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


def accuracy(X: jnp.array, y: jnp.array, params: jnp.array, quantum_model: callable) -> jnp.ndarray:
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


def TestAcc(X: jnp.array, y: jnp.array, optimal_params: jnp.array, quantum_model: callable) -> tuple:
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