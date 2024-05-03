import jax
from jax import numpy as jnp
import optax
from sklearn.metrics import accuracy_score
from qcnn_architecture import QCNNArchitecture
import pennylane as qml
from config import dev, wires, lr


@qml.qnode(device=dev, interface="jax")
def qcnn(data: jnp.array, params: jnp.array) -> qml.probs:
    circuit = QCNNArchitecture(device=dev, wires=wires)
    qml.AmplitudeEmbedding(features=data, wires=range(len(wires)), normalize=True, pad_with=0.)
    circuit.QCNN(params)
    probs = qml.probs(wires=[3, 5])
    return probs


def SelectModel(circuit_function: callable) -> jax.jit:
    qnn_teacher = jax.vmap(
        circuit_function,
        (0, None)
    )
    quantum_model = jax.jit(qnn_teacher)
    return quantum_model


def optax_bce(x: jnp.array, y: jnp.array, theta: jnp.array, quantum_model: callable) -> jnp.ndarray:
    labels = jnp.array(y)
    pred = jnp.array(quantum_model(x, theta))
    one_hot = jax.nn.one_hot(labels, pred.shape[1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=pred, labels=one_hot))
    return loss


def optimizer_update(opt_state: optax, params: jnp.array, x: jnp.array, y: jnp.array, quantum_model: callable) -> tuple:
    optimizer = optax.adam(learning_rate=lr)
    loss_value, grads = jax.value_and_grad(lambda theta: optax_bce(x, y, theta,quantum_model))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


def accuracy(X: jnp.array, y: jnp.array, params: jnp.array, quantum_model: callable) -> jnp.ndarray:
    labels = jnp.array(y)
    pred = jnp.array(quantum_model(X, params))
    accuracy = jnp.sum(jnp.argmax(jnp.array(pred), axis=1) == labels)
    return accuracy/len(labels)


def TestAcc(X: jnp.array, y: jnp.array, optimal_params: jnp.array, quantum_model: callable) -> tuple:
    labels = jnp.array(y)
    predictions = quantum_model(X, optimal_params)
    estimation = jnp.argmax(predictions, axis=1)
    return estimation, accuracy_score(y_true=labels, y_pred=estimation)