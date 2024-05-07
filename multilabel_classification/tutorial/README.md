__Quantum multilabel classification with JAX__

__Author:__ Francesco Aldo Venturelli;
__Date created:__ 2024/07/05;
__Description:__ Quantum convolutional neural network (QCNN) for multilabel image classification written by combining Pennylane with JAX. 
In this simple tutorial we select four different classes of digits images from the *sklearn.datasets.load_digits* and we construct a quantum convolutional neural network to train and make the classification of multiclass images.
After model training, we select the last updated and optimal parameters (which correspond to the maximum value of the validation accuracy) and use them to test the model on the unseen test images. To have an idea about all the steps needed to complete the experiment we recommend to have a look at the scripts in the */src* folder. Hope this could be useful for you, feel free to use these codes and make further improvements. With this code we would like to emphasize the ability of making multilabel classification and stop of being constrained by binary classifications!


__Goal:__ Train a fully quantum convolutional neural network to classify four classes of images using train and validation sets. The idea behind the project, is also the realize a multilabel classification problem, by measuring two distinct qubits, and to exploit JAX's speed in vectorizing the training.


__Installing libraries:__
`!pip install numpy==1.23.5`
`!pip install scikit-learn==1.3.0`
`!pip install jax==0.4.8`
`!pip install optax==0.1.5`
`!pip install matplotlib==3.7.1`
`!pip install pennyLane==0.30.0`


__Project's structure:__
── tutorial
│   ├── run_tutorial.ipynb
│   ├── config.py
│   ├── src
│   │   ├── __init__.py
│   │   ├── plot_results.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   └── qcnn_architecture.py
│   └── dataset
│       └── digits.py
