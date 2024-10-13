import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
import matplotlib.pyplot as plt

# Got inspiration from engine.py file of the following github repo:
# https://github.com/shakes76/GFNet
# And my train/evaluate code from the brain GAN code.

# This class has been taken from the github repo mentioned above, and
# adjusted to fit my purposes.
class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

# def train_model(model, train_loader, val_loader, num_epochs=25):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     train_loss, val_loss, train_acc, val_acc = [], [], [], []

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / len(train_loader)
#         epoch_acc = correct / total
        
#         train_loss.append(epoch_loss)
#         train_acc.append(epoch_acc)
        
#         # Validation step (similar to training loop but without optimizer step)
#         val_epoch_loss, val_epoch_acc = validate_model(model, val_loader, criterion)
#         val_loss.append(val_epoch_loss)
#         val_acc.append(val_epoch_acc)
        
#         print(f"Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_acc}")

#     plot_metrics(train_loss, val_loss, train_acc, val_acc)
#     torch.save(model.state_dict(), "gfnet_model.pth")

# def validate_model(model, val_loader, criterion):
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     avg_loss = val_loss / len(val_loader)
#     accuracy = correct / total
#     return avg_loss, accuracy

# def plot_metrics(train_loss, val_loss, train_acc, val_acc):
#     plt.figure(figsize=(10, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss, label='Train Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.title('Loss Over Time')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc, label='Train Accuracy')
#     plt.plot(val_acc, label='Validation Accuracy')
#     plt.title('Accuracy Over Time')
#     plt.legend()
    
#     plt.show()
#     plt.savefig("/home/Student/s4743000/COMP3710/PatternAnalysis-2024/recognition/ADNI-GFNet-47430004/test/train/test_train", bbox_inches='tight', pad_inches=0)
#     plt.close()
