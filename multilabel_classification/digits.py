import jax.numpy as jnp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


seed = 5
klass = (0, 1, 2, 3)


def normalize_data(data: np.array) -> np.array:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def select_binary_classes(data: np.array, target: np.array, classes: tuple) -> tuple:
    """
    Selects samples from the dataset that belong to the specified binary classes.

    Args:
    - data (numpy.ndarray): Input data.
    - target (numpy.ndarray): Target labels.
    - classes (tuple): Tuple containing the classes to select.

    Returns:
    - numpy.ndarray: Selected data samples.
    - numpy.ndarray: Corresponding target labels.
    """
    mask = np.isin(target, classes)
    selected_data = data[mask]
    selected_target = target[mask]
    return selected_data, selected_target


digits = load_digits()
X, y = digits.data, digits.target

X_binary, y_binary = select_binary_classes(X, y, classes=klass)
X_binary_normalized = normalize_data(X_binary)
X_train, X_val, y_train, y_val = train_test_split(X_binary_normalized, y_binary, random_state=seed, test_size=0.5)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, random_state=seed, test_size=0.2)

X_train = jnp.asarray(X_train)
y_train = jnp.asarray(y_train)
X_test = jnp.asarray(X_test)
y_test = jnp.asarray(y_test)
X_val = jnp.asarray(X_val)
y_val = jnp.asarray(y_val)

def Info() -> None:
    print("General information about the training set:")
    print("Classes:", klass)
    print("Number of training samples:", X_train.shape[0])
    print("Number of testing samples:", X_test.shape[0])
    print("Number of each training class:",
          {
              klass[0]: np.count_nonzero(y_train == klass[0]),
              klass[1]: np.count_nonzero(y_train == klass[1]),
              klass[2]: np.count_nonzero(y_train == klass[2]),
              klass[3]: np.count_nonzero(y_train == klass[3])})

    print("Number of each testing class:",
          {
              klass[0]: np.count_nonzero(y_test == klass[0]),
              klass[1]: np.count_nonzero(y_test == klass[1]),
              klass[2]: np.count_nonzero(y_test == klass[2]),
              klass[3]: np.count_nonzero(y_test == klass[3])
          })


if "__main__" == __name__:
    Info()
