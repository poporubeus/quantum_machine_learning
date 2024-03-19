from jax import numpy as jnp
import jax
import numpy as np
from sklearn.utils import gen_batches
from keras.datasets import mnist
from load_data import down_sample
import optax
from sklearn.metrics import accuracy_score


def make_test(classes_of_items, new_size, test_seed):
    num_samples_per_class = 300

    (_, _), (X_test, y_test) = mnist.load_data()

    mask = np.isin(y_test, classes_of_items)
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]

    X_test_new = np.asarray([down_sample(x_test, new_size) for x_test in X_test_filtered])
    X_test_final = np.array(X_test_new).reshape((X_test_new.shape[0]), -1) / 255.0
    y_test_final = np.array(y_test_filtered)

    selected_indices = []
    for class_idx in classes_of_items:
        class_indices = np.where(y_test_filtered == class_idx)[0][:int(num_samples_per_class / 2)]
        selected_indices.extend(class_indices)

    # Shuffle selected indices
    np.random.shuffle(selected_indices)
    X_test_final = X_test_final[selected_indices]
    y_test_final = y_test_final[selected_indices]
    X_test_final = jnp.asarray(X_test_final)
    y_test_final = jnp.asarray(y_test_final)
    return (X_test_final,
            y_test_final)



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

def calculate_accuracy(X, y, params):
    labels = jnp.array(y)
    predictions = jnp.array(qnn(X, params))
    predicted_labels = jnp.array(jnp.argmax(predictions, axis=1))
    accuracy = jnp.mean(accuracy_score(labels, predicted_labels))
    return accuracy






def training(qnn_batched, lys):
    qnn = jax.jit(qnn_batched)
    key = jax.random.PRNGKey(seed)
    initial_params = jax.random.normal(key, shape=((param_per_gate * n_qubits + param_per_gate * entangling_gate +
                                                    param_per_gate * n_qubits) * lys,))
    key = jax.random.split(key)[0]
    params = jnp.copy(initial_params)
    opt_state = optimizer.init(initial_params)

    train_cost_epochs, train_acc_epochs, val_cost_epochs, val_acc_epochs = [], [], [], []

    for epoch in range(1, n_epochs + 1):
        idxs_dataset = jax.random.choice(key, jnp.array(list(range(X_train.shape[0]))), shape=(X_train.shape[0],),
                                         replace=False)
        key = jax.random.split(key)[0]
        for i in gen_batches(X_train.shape[0], batch_size):
            idxs = idxs_dataset[i]
            params, opt_state, cost = optimizer_update(opt_state, params, X_train[idxs, :], y_train[idxs])
            '''params, opt_state, cost = optimizer_update(opt_state, params, X_train, y_train)'''
        cost = binary_crossentropy(X_train, y_train, params, )
        val_cost = binary_crossentropy(X_val, y_val, params, )
        train_cost_epochs.append(cost)
        val_cost_epochs.append(val_cost)
        # Accuracy during training and validation
        train_acc = calculate_accuracy(X_train, y_train, params, )
        val_acc = calculate_accuracy(X_val, y_val, params, )
        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)
        print(f"Epoch: {epoch}, ---Train loss: ", cost, "---Val loss: ", val_cost, "---Train acc:", train_acc,
              "---Val acc.:", val_acc)
        optimal_params = params  # Optimal parameters after training
        path = "/Users/francescoaldoventurelli/Desktop/"
        np.save(path + "trained_weight_file_1layer.npy", optimal_params)
    return (
        train_cost_epochs,
        val_cost_epochs,
        train_acc_epochs,
        val_acc_epochs,
        optimal_params
    )






