import pennylane as qml
import jax
from jax import numpy as jnp
from sklearn.utils import gen_batches
import matplotlib.pyplot as plt
import optax
from sklearn.metrics import accuracy_score
from digits import X_train, y_train, X_test, y_test, X_val, y_val
from qcnn_architecture import QCNNArchitecture


lr = 0.001
seed = 2222
wires = list(range(6))
dev = qml.device("default.qubit", wires=6)


@qml.qnode(device=dev, interface="jax")
def qcnn(data, params):
    circuit = QCNNArchitecture(device=dev, wires=wires)
    qml.AmplitudeEmbedding(features=data, wires=range(len(wires)), normalize=True, pad_with=0.)
    circuit.QCNN(params)
    probs = qml.probs(wires=[3, 5])
    return probs


def optax_bce(x,y,theta, quantum_model):
    labels = jnp.array(y)
    pred = jnp.array(quantum_model(x, theta))
    one_hot = jax.nn.one_hot(labels, pred.shape[1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=pred, labels=one_hot))
    return loss


def optimizer_update(opt_state, params, x, y, quantum_model):
    optimizer = optax.adam(learning_rate=lr)
    loss_value, grads = jax.value_and_grad(lambda theta: optax_bce(x, y, theta,quantum_model))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


def accuracy(X, y, params, quantum_model):
    labels = jnp.array(y)
    pred = jnp.array(quantum_model(X, params))
    accuracy = jnp.sum(jnp.argmax(jnp.array(pred), axis=1) == labels)
    return accuracy/len(labels)


def training(seed, model, selected_shape):
    n_epochs = 150
    batch_size = 32
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
        # Accuracy during training and validation
        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)
        print(f"Epoch: {epoch}, ---Train loss: ", cost, "---Train acc: ", train_acc,
              "| ---Val loss: ", val_cost, "---Val acc: ", val_acc)
    optimal_params = params  # Optimal parameters after training saved here, ready to be used for predictions.
    return [train_cost_epochs, train_acc_epochs, val_cost_epochs, val_acc_epochs, optimal_params]


def TestAcc(X, y, optimal_params, quantum_model):
    labels = jnp.array(y)
    predictions = quantum_model(X, optimal_params)
    estimation = jnp.argmax(predictions, axis=1)
    return accuracy_score(y_true=labels, y_pred=estimation)


def SelectModel(circuit_function: callable) -> jax.jit:
    qnn_teacher = jax.vmap(circuit_function,
                           (0, None))  # use vmap to apply quantum_nn function to a vector, it's typical in JAX.
    quantum_model = jax.jit(qnn_teacher)
    return quantum_model


quantum_model = SelectModel(qcnn)
train_cost, train_acc, val_cost, val_acc, optimal_params = training(seed=seed, model=quantum_model, selected_shape=60)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4.5))
axs[0].plot(train_cost, label="Train cost teacher", color='royalblue')
axs[0].plot(val_cost, label="Validation cost teacher", color='tab:blue')
axs[1].plot(train_acc, label='Train accuracy teacher', color='orangered')
axs[1].plot(val_acc, label='Validation accuracy teacher', color='orange')
axs[0].set_xlabel("Epoch")
axs[1].set_xlabel("Epoch")
axs[0].set_xticks(jnp.arange(0, 160, 10))
axs[1].set_xticks(jnp.arange(0, 160, 10))
axs[0].legend()
axs[1].legend()
plt.show()


print("Test acc. teacher model:", TestAcc(X_test, y_test, optimal_params=optimal_params, quantum_model=quantum_model))
