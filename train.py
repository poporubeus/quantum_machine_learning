import optax
from sklearn.utils import gen_batches
from load_data import RandomMnist
import pennylane as qml
from quantum_model import quantum_neural_network
from sklearn.metrics import accuracy_score
import jax
from jax import numpy as jnp

lr = 0.01
n_qubits = 8
layers = 2
n_epochs = 500
batch_size = 16
seed = 123
param_per_gate = 3
entangling_gate = 7
new_shape = 8
dataset = RandomMnist(classes_of_items=[0, 1], num_train_samples=500, shuffle=True, resize=new_shape, my_seed=999)
X_train, y_train, X_val, y_val = dataset.data()


dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface='jax')
def quantum_nn(X, thetas):
    model = quantum_neural_network(n_wires=n_qubits)
    params_per_sublayer = 69
    for layer in range(layers):
        model.QuantumMap(X)
        qml.Barrier()
        j = layer * params_per_sublayer
        model.QuantumRing_layer(thetas[j:j+(params_per_sublayer)],)
        j += (params_per_sublayer)*(layer)
        qml.Barrier(wires=range(n_qubits),only_visual=True)
    qml.Barrier(wires=range(n_qubits),only_visual=True)
    return qml.probs(wires=0)

@jax.jit
def binary_crossentropy(x, y, theta):
    labels = jnp.array(y)
    pred = jnp.array(qnn(x, theta))
    loss = 0
    for l, p in zip(labels, pred):
        loss += l * (jnp.log(p[l])) + (1 - l) * jnp.log(1 - p[1 - l])
    return -jnp.sum(loss)/len(labels)

@jax.jit
def optimizer_update(opt_state, params, x, y):
    loss_value, grads = jax.value_and_grad(lambda theta: binary_crossentropy(x, y, theta))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

@jax.jit
def calculate_accuracy(X, y, params):
    labels = jnp.array(y)
    predictions = jnp.array(qnn(X, params))
    predicted_labels = jnp.array(jnp.argmax(predictions, axis=1))
    accuracy = jnp.mean(accuracy_score(labels, predicted_labels))
    return accuracy


qnn_batched = jax.vmap(quantum_nn, (0, None))
qnn = jax.jit(qnn_batched)
optimizer = optax.adam(learning_rate=lr)
def training_process(param_per_gate, entangling_gate, layers, n_qubits, seed, n_epochs, X_train, y_train, X_val, y_val, batch_size):
    key = jax.random.PRNGKey(seed)
    initial_params = jax.random.normal(key, shape=((param_per_gate * n_qubits + param_per_gate * entangling_gate +
                                                    param_per_gate * n_qubits) * layers,))
    key = jax.random.split(key)[0]
    params = jnp.copy(initial_params)
    opt_state = optimizer.init(initial_params)

    train_cost_epochs, train_acc_epochs, val_cost_epochs, val_acc_epochs = [], [], [], []

    for epoch in range(1,n_epochs+1):
        idxs_dataset = jax.random.choice(key, jnp.array(list(range(X_train.shape[0]))), shape=(X_train.shape[0],), replace=False)
        key = jax.random.split(key)[0]
        for i in gen_batches(X_train.shape[0], batch_size):
            idxs = idxs_dataset[i]
            params, opt_state, cost = optimizer_update(opt_state, params, X_train[idxs, :], y_train[idxs])
        cost = binary_crossentropy(X_train, y_train, params,)
        val_cost = binary_crossentropy(X_val, y_val, params,)
        train_cost_epochs.append(cost)
        val_cost_epochs.append(val_cost)
        # Accuracy during training and validation
        train_acc = calculate_accuracy(X_train, y_train, params,)
        val_acc = calculate_accuracy(X_val, y_val, params,)
        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)
        print(f"Epoch: {epoch}, ---Train loss: ", cost, "---Val loss: ", val_cost, "---Train acc:", train_acc, "---Val acc:", val_acc)
        optimal_params = params
        return (
            train_cost_epochs,
            val_cost_epochs,
            train_acc_epochs,
            val_acc_epochs,
            optimal_params
        )


loss_train, loss_val, acc_train, acc_val, optimal_params = training_process(param_per_gate, entangling_gate,
                                                                            layers, n_qubits, seed, n_epochs,
                                                                            X_train, y_train, X_val, y_val, batch_size)