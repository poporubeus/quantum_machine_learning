from distillation import cpu_device
from Utility import test, train
from data_knowledge import train_loader, test_loader
from quantum_student_model import LightNN, n_qubits, n_layers


nn_light = LightNN(n_layers, n_qubits).to(device=cpu_device)
print("Train quantum LIGHTER model...")
train(nn_light, train_loader, epochs=50, learning_rate=0.001, seed=999, device=cpu_device)
test_accuracy_light = test(nn_light, test_loader, device=cpu_device)