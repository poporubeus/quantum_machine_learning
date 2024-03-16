import numpy as np
from jax import numpy as jnp
import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


def down_sample(x_array: np.ndarray, size: int) -> np.ndarray:
    """
    Function that reduces size of the images array.
    Args:
        x_array (array): image array.
        size (int): new size to resize.
    Returns:
        new_array (array): resized image.
    """
    new_size = (size, size)
    x_array = np.reshape(x_array, (x_array.shape[0], x_array.shape[1], 1))
    new_array = tf.image.resize(x_array, new_size)
    return new_array.numpy()

class RandomMnist:
    """
    Class that creates a binary dataset out of Mnist after specifying the classes of images to deal with,
    based on Keras's API.
    The dataset created here is composed by train and validation set.
    Args:
        classes_of_items (list): classes of images;
        num_train_samples (int): number of training images;
        shuffle (bool): if True the images are shuffled before creating the dataset;
        resize (int): new image dimension (it must be lower than 28);
        my_seed (int): seed for reproducibility.
    Returns:
        X_train (array): set of training images of shape (num_images, new_shape*new_shape);
        y_train (array): set of training labels of shape (num_images,);
        X_val (array): set of validation images of shape (num_images, new_shape*new_shape);
        y_val (array): set of validation labels of shape (num_images,).
    Note that the number of validation data is 0.2*num_train_samples.
    """

    def __init__(self, classes_of_items: list, num_train_samples: int, shuffle: bool,
                 resize: int, my_seed: int) -> None:
        self.classes_of_items = classes_of_items
        self.num_train_samples = num_train_samples
        self.shuffle = shuffle
        self.resize = resize
        self.my_seed = my_seed

    def data(self) -> tuple:
        np.random.seed(self.my_seed)
        (train_X, train_y), (_, _) = mnist.load_data()
        X_train_filtered = train_X[np.isin(train_y, [self.classes_of_items[0], self.classes_of_items[1]])]
        y_train_filtered = train_y[np.isin(train_y, [self.classes_of_items[0], self.classes_of_items[1]])]

        X_train_filtered = X_train_filtered.astype('float16') / 255
        X_train_new = []
        if self.resize is not None and self.resize <= 28:
            for train in X_train_filtered:
                X_train_new.append(down_sample(train, self.resize))
        else:
            raise Exception("The new size must be smaller than the actual Mnist size that is 28!")
        # shuffle
        X_train_new = np.array(X_train_new)
        if self.shuffle:
            shuffled_indices = np.arange(len(X_train_new))
            np.random.shuffle(shuffled_indices)
            X_train_new = X_train_new[shuffled_indices]
            y_train_filtered = y_train_filtered[shuffled_indices]

        if self.num_train_samples is not None:
            num_samples_per_class = self.num_train_samples // len(self.classes_of_items)
            selected_indices = []
            for class_idx in self.classes_of_items:
                class_indices = np.where(y_train_filtered == class_idx)[0][:num_samples_per_class]
                selected_indices.extend(class_indices)
            X_train_ = X_train_new[selected_indices]
            y_train_filtered = y_train_filtered[selected_indices]
        else:
            raise RuntimeError("Insert number of images to form the train set.")

        X_train_ = X_train_.reshape(X_train_.shape[0], X_train_.shape[1] * X_train_.shape[2])
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_, y_train_filtered, test_size=0.2, random_state=42)

        return (
            jnp.asarray(X_train_final),
            jnp.asarray(y_train_final),
            jnp.asarray(X_val),
            jnp.asarray(y_val),
        )