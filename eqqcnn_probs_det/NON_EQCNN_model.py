import pennylane as qml
import numpy as np
import jax
from jax import numpy as jnp
import optax
from probabilistic_QCNN_DATASET import new_x_train, Y_train, new_x_test, Y_test
from sklearn.utils import gen_batches
from sklearn.metrics import accuracy_score


total_wires = 8
device = qml.device("default.qubit", wires=total_wires)

def ConvFilter(weigths, wires):
    qml.U3(weigths[0], weigths[1], weigths[2], wires=wires[0])
    qml.RX(weigths[3], wires=wires[1])
    #qml.U3(weigths[3], weigths[4], weigths[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def Poollayer(weigths, wires):
    qml.CRZ(weigths[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(weigths[1], wires=[wires[0], wires[1]])


@qml.qnode(device = device, interface="jax")
def nonEQCNN(data, params):
    fm = qml.AmplitudeEmbedding(features=data, wires=range(8), normalize=True, pad_with=0.)
    ConvFilter(params[:4], wires=[0, 1])
    ConvFilter(params[:4], wires=[2, 3])
    ConvFilter(params[:4], wires=[4, 5])
    ConvFilter(params[:4], wires=[6, 7])
    ConvFilter(params[:4], wires=[1, 2])
    ConvFilter(params[:4], wires=[3, 4])
    ConvFilter(params[:4], wires=[5, 6])
    ConvFilter(params[:4], wires=[7, 0])
    Poollayer(params[4:6], wires=[1, 0])
    Poollayer(params[4:6], wires=[3, 2])
    Poollayer(params[4:6], wires=[5, 4])
    Poollayer(params[4:6], wires=[7, 6])
    # 2nd layer
    ConvFilter(params[6:10], wires=[0, 2])
    ConvFilter(params[6:10], wires=[4, 6])
    ConvFilter(params[6:10], wires=[2, 4])
    ConvFilter(params[6:10], wires=[6, 0])
    Poollayer(params[10:12], wires=[2, 0])
    Poollayer(params[10:12], wires=[6, 4])
    # 3rd layer
    ConvFilter(params[12:16], wires=[0, 4])
    Poollayer(params[16:18], wires=[4, 0])
    return qml.probs(wires=0)


def SelectModel(circuit_function: callable) -> jax.jit:
    """
    Select the model to use in the experiment. Note that any model is allowed to be used
    within this experiment, but it needs to have the same parameters as the circuit function.
    :param circuit_function: (callable) The function which generates the circuit.
    :return: quantum_model (jax.jit) The quantum circuit mapped and executed.
    """
    qnn_teacher = jax.vmap(
        circuit_function,
        (0, None)
    )
    quantum_model = jax.jit(qnn_teacher)
    return quantum_model


def optax_bce(x: jnp.array, y: jnp.array, theta: jnp.array, quantum_model: callable) -> jnp.ndarray:
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
    one_hot = jax.nn.one_hot(labels, pred.shape[1])
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(pred, one_hot))
    # loss = jnp.mean(optax.softmax_cross_entropy(logits=pred, labels=one_hot))
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
    optimizer = optax.adam(learning_rate=0.001)
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
    mean_acc = accuracy / len(labels)
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


new_x_train, new_x_test = jnp.asarray(new_x_train), jnp.asarray(new_x_test)
Y_train, Y_test = jnp.asarray(Y_train), jnp.asarray(Y_test)


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
    path = "/home/fv/storage1/qml/non_eqcnn"
    f = open(path + f'/mnist_nonqecnn_{str(seed)}.txt', 'w')
    key = jax.random.PRNGKey(seed)
    initial_params = jax.random.normal(key, shape=(selected_shape,))
    key = jax.random.split(key)[0]
    params = jnp.copy(initial_params)
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(initial_params)
    train_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], []
    for epoch in range(1, n_epochs + 1):
        idxs_dataset = jax.random.choice(key, jnp.array(list(range(new_x_train.shape[0]))),
                                         shape=(new_x_train.shape[0],),
                                         replace=False)
        key = jax.random.split(key)[0]
        for i in gen_batches(new_x_train.shape[0],
                             batch_size):
            idxs = idxs_dataset[i]
            params, opt_state, cost = optimizer_update(opt_state, params, new_x_train[idxs, :], Y_train[idxs], model)
        cost = optax_bce(new_x_train, Y_train, params, model)
        # test_cost = optax_bce(new_x_test, y_val, params, model)
        train_acc = accuracy(new_x_train, Y_train, params, model)
        test_acc = accuracy(new_x_test, Y_test, params, model)
        train_cost_epochs.append(cost)
        # val_cost_epochs.append(val_cost)
        train_acc_epochs.append(train_acc)
        test_acc_epochs.append(test_acc)
        # val_acc_epochs.append(val_acc)
        print(f"Epoch: {epoch}, ---Train loss: ", cost, "---Train acc: ", train_acc,
              "---Test acc: ", test_acc
              )
        f.write(str(epoch) + "  " + str(cost) + "  " + str(train_acc) + "  " + str(test_acc))
        f.write("\n")
    optimal_params = params  # Optimal parameters after training saved here, ready to be used for predictions.
    f.close()
    return [train_cost_epochs, train_acc_epochs, test_acc_epochs, f]


epoch = 30
seed_list = np.arange(start=1, stop=21, step=1)
run_number = np.arange(start=1, stop=21, step=1)
loss_final, acc_final = [], []
test_acc_final = []
model = SelectModel(nonEQCNN)
for i in range(len(seed_list)):
    print(f"Run number: {run_number[i]}")
    loss, acc, test_acc, file = training(
        seed=seed_list[i],
        model=model,
        selected_shape=18,
        n_epochs=epoch,
        batch_size=32
    )
    loss_final.append(loss)
    acc_final.append(acc)
    test_acc_final.append(test_acc)
    # test_acc.append(TestAcc(new_x_test, Y_test, optimal_params=opt_params, quantum_model=model))
loss_final_mean, loss_final_std = np.mean(loss_final), np.std(loss_final)
acc_final_mean, acc_final_std = np.mean(acc_final), np.std(acc_final)
test_acc_mean = np.mean(test_acc_final), np.std(test_acc_final)