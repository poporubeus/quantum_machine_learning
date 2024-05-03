from distillation_wout_T import (train_knowledge_distillation,
                          cpu_device, nn_deep)
from Utility import test, train
from data_knowledge import train_loader, test_loader
import numpy as np
from quantum_student_model import LightNN, n_qubits, n_layers


nn_light = LightNN(n_layers, n_qubits).to(device=cpu_device)
print("Train quantum LIGHTER model...")
train(nn_light, train_loader, epochs=20, learning_rate=0.001, seed=999, device=cpu_device)
test_accuracy_light = test(nn_light, test_loader, device=cpu_device)

new_nn_light = LightNN(n_layers, n_qubits).to(device=cpu_device)
weights_loss = [0.25, 0.4, 0.5]
test_w_different_weights = np.zeros(len(weights_loss))
for s, i in enumerate(range(len(weights_loss))):
    print(f"Train quantum LIGHTER model with KNOWLEDGE-DISTILLATION...with weight{i+1}")
    train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=50,
                                 learning_rate=0.001, soft_target_loss_weight=weights_loss[i], ce_loss_weight=1-weights_loss[i],
                                 device=cpu_device, seed=s)
    test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device=cpu_device)
    test_w_different_weights[i] = test_accuracy_light_ce_and_kd
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

print("Test quantum lighter model NOT DISTILLED: \n", test_accuracy_light)
print("")
print("Test quantum lighter models DISTILLED: \n", test_w_different_weights)