from Utility import train
import torch
from classical_model import DeepNN
from torch import nn
from torch import optim
from data_knowledge import train_loader


#path = "/Users/francescoaldoventurelli/Desktop/"
cpu_device = torch.device("cpu")


def t_process(out, Temperature):
    return out / Temperature


nn_deep = DeepNN().to(device=cpu_device)
print("Train classical LARGE model...")
train(model=nn_deep, train_loader=train_loader, epochs=10, learning_rate=0.001, seed=999, device=cpu_device)
#torch.save(nn_deep.state_dict(), path + "model_deep.pt")


def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, Temperature, soft_target_loss_weight,
                                 ce_loss_weight, device, seed):
    torch.manual_seed(seed)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    total_correct = 0
    total_instances = 0
    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode
    print("Start training distilled model")

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                #teacher_logits = teacher(inputs.to('cpu'))
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = t_process(student(inputs), Temperature=Temperature)
            soft_targets = nn.functional.softmax(teacher_logits / Temperature, dim=-1)
            #soft_prob = torch.log_softmax(student_logits, dim=-1)  ## I copied from tutorial
            # but they are already probabilities!!!

            #soft_prob = torch.log(student_logits)
            soft_prob = torch.log(student_logits)

            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (
                        Temperature ** 2)

            label_loss = ce_loss(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                classifications = torch.argmax(student(images), dim=1)
                correct_predictions = sum(classifications == labels).item()
                total_correct += correct_predictions
                total_instances += len(images)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}," 
              f"Accuracy: {(total_correct / total_instances) * 100:.2f}%")