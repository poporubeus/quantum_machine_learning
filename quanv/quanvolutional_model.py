import pennylane as qml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from digits import select_binary_classes, normalize_data, load_digits, train_test_split
import matplotlib.pyplot as plt


img_size = 8
seed = 8888
np.random.seed(seed)
tf.random.set_seed(seed)
n_layers = 3
q_bits = 4
n_epochs = 200
dev = qml.device("default.qubit", wires=q_bits)
first_qconv_params = np.random.uniform(high=2*np.pi, size=(n_layers, q_bits))
second_qconv_params = np.random.uniform(high=2*np.pi, size=(n_layers, q_bits))


digits = load_digits()
X, y = digits.data, digits.target
klass = (0, 1, 2, 3)
X_binary, y_binary = select_binary_classes(X, y, classes=klass)
X_binary_normalized = normalize_data(X_binary)
X_train, X_val, y_train, y_val = train_test_split(X_binary_normalized, y_binary, random_state=seed, test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state=seed, test_size=0.2)
train_images = np.array(X_train[..., tf.newaxis])
test_images = np.array(X_test[..., tf.newaxis])
val_images = np.array(X_val[..., tf.newaxis])
train_images = np.reshape(train_images, newshape=(train_images.shape[0], 8, 8, 1))
test_images = np.reshape(test_images, newshape=(test_images.shape[0], 8, 8, 1))
val_images = np.reshape(val_images, newshape=(val_images.shape[0], 8, 8, 1))
train_labels = y_train
test_labels = y_test
val_labels = y_val
print(test_labels.shape)


@qml.qnode(dev, interface="tensorflow")
def circuit(phi):
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)
    qml.RandomLayers(first_qconv_params, wires=list(range(q_bits)))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]


@qml.qnode(dev, interface="tensorflow")
def circuit2(phi):
    for j in range(4):
        qml.RX(np.pi * phi[j], wires=j)
    qml.RandomLayers(second_qconv_params, wires=list(range(q_bits)))
    return [qml.expval(qml.PauliZ(wires=j)) for j in range(4)]


def quanv(image):
    out = np.zeros((4, 4, 4))
    final_out = np.zeros((2, 2, 4))
    for j in range(0, img_size, 2):
        for k in range(0, img_size, 2):
            q_results = circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
        img_size_new = img_size // 2
        out = np.reshape(out, (out.shape[0], 4, 4, 1))
        for j in range(0, img_size_new, 2):
            for k in range(0, img_size_new, 2):
                q_results_final = circuit2(
                    [
                        out[j, k, 0],
                        out[j, k + 1, 0],
                        out[j + 1, k, 0],
                        out[j + 1, k + 1, 0]
                    ]
                )
                for c in range(4):
                    final_out[j // 2, k // 2, c] = q_results_final[c]
    return final_out


def quanvSingle(image):
    out = np.zeros((4, 4, 4))
    for j in range(0, img_size, 2):
        for k in range(0, img_size, 2):
            q_results = circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


q_train_images = []
print("Quantum pre-processing of train images:")
for idx, img in enumerate(train_images):
    q_train_images.append(quanv(img))
q_train_images = np.asarray(q_train_images)

q_val_images = []
print("\nQuantum pre-processing of validation images:")
for idx, img in enumerate(val_images):
    q_val_images.append(quanv(img))
q_val_images = np.asarray(q_val_images)

'''q_test_images = []
print("\nQuantum pre-processing of test images:")
for idx, img in enumerate(test_images):
    q_test_images.append(quanv(img))
q_test_images = np.asarray(q_test_images)'''


def Model():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation="softmax")
    ])

    '''model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=keras.metrics.Accuracy()
    )'''
    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


q_model = Model()
q_history = q_model.fit(
    q_train_images,
    train_labels,
    validation_data=(q_val_images, val_labels),
    batch_size=32,
    epochs=n_epochs,
    verbose=2,
)


q_train_images2 = []
print("Quantum pre-processing of train images w only 1 quanv:")
for idx, img in enumerate(train_images):
    q_train_images2.append(quanvSingle(img))
q_train_images2 = np.asarray(q_train_images2)

q_val_images2 = []
print("\nQuantum pre-processing of validation images w only 1 quanv:")
for idx, img in enumerate(val_images):
    q_val_images2.append(quanvSingle(img))
q_val_images2 = np.asarray(q_val_images2)


q_model2 = Model()
q_history2 = q_model2.fit(
    q_train_images2,
    train_labels,
    validation_data=(q_val_images2, val_labels),
    batch_size=32,
    epochs=n_epochs,
    verbose=2,
)


c_model = Model()

history_c = c_model.fit(
    train_images,
    train_labels,
    validation_data=(val_images, val_labels),
    batch_size=32,
    epochs=n_epochs,
    verbose=2,
)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(q_history.history["val_accuracy"], color="royalblue", label="With 2 quantum layers")
ax1.plot(q_history2.history["val_accuracy"], color="green", label="With 1 quantum layer")
ax1.plot(history_c.history["val_accuracy"], color="orange", label="CNN")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(q_history.history["val_loss"], color="royalblue", label="With 2 quantum layers")
ax2.plot(q_history2.history["val_loss"], color="green", label="With 1 quantum layer")
ax2.plot(history_c.history["val_loss"], color="orange", label="CNN")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.show()