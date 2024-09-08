import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import time
from collections import OrderedDict

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions and classes
def node_def(module, **kwargs):
    return lambda *args, **kw: module(*args, **{**kwargs, **kw})

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Add(nn.Module):
    def forward(self, x, y):
        return x + y

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def forward(self, x):
        return x * self.weight

class Identity(nn.Module):
    def forward(self, x):
        return x

# Define building blocks
conv = node_def(nn.Conv2d)
linear = node_def(nn.Linear)
batch_norm = node_def(nn.BatchNorm2d)
pool = node_def(nn.MaxPool2d)
relu = node_def(nn.ReLU)

def conv_block(c_in, c_out):
    return OrderedDict([
        ('conv', conv(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
        ('norm', batch_norm(c_out)),
        ('act', relu()),
    ])

def conv_pool_block(c_in, c_out):
    return OrderedDict(list(conv_block(c_in, c_out).items()) + [('pool', pool(2))])

def conv_pool_block_pre(c_in, c_out):
    return OrderedDict([
        ('conv', conv(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
        ('pool', pool(2)),
        ('norm', batch_norm(c_out)),
        ('act', relu()),
    ])

def residual(c, conv_block):
    return OrderedDict([
        ('in', Identity()),
        ('res1', nn.Sequential(conv_block(c, c))),
        ('res2', nn.Sequential(conv_block(c, c))),
        ('out', Identity()),
        ('add', Add()),
    ])

def build_network(channels, extra_layers, res_layers, scale, conv_block=conv_block,
                  prep_block=conv_block, conv_pool_block=conv_pool_block):
    net = OrderedDict([
        ('prep', nn.Sequential(prep_block(3, channels['prep']))),
        ('layer1', nn.Sequential(conv_pool_block(channels['prep'], channels['layer1']))),
        ('layer2', nn.Sequential(conv_pool_block(channels['layer1'], channels['layer2']))),
        ('layer3', nn.Sequential(conv_pool_block(channels['layer2'], channels['layer3']))),
        ('pool', pool(4)),
        ('flatten', Flatten()),
        ('classifier', linear(channels['layer3'], 10, bias=False)),
        ('scale', Mul(scale)),
    ])
    
    for layer in res_layers:
        current_layer = net[layer]
        net[layer] = nn.Sequential(OrderedDict([
            ('main', current_layer),
            ('residual', nn.Sequential(residual(channels[layer], conv_block)))
        ]))
    
    for layer in extra_layers:
        current_layer = net[layer]
        net[layer] = nn.Sequential(OrderedDict([
            ('main', current_layer),
            ('extra', nn.Sequential(conv_block(channels[layer], channels[layer])))
        ]))
    
    return nn.Sequential(net)

# Define the neural network
# class FastCIFAR10Net(nn.Module):
#     def __init__(self):
#         super(FastCIFAR10Net, self).__init__()
#         channels = {
#             'prep': 64,
#             'layer1': 128,
#             'layer2': 256,
#             'layer3': 512,
#         }
#         self.network = build_network(
#             channels,
#             extra_layers=['layer1', 'layer2'],
#             res_layers=['layer1', 'layer2', 'layer3'],
#             scale=1.0,
#             conv_block=conv_block,
#             conv_pool_block=conv_pool_block
#         )

#     def forward(self, x):
#         return self.network(x)

class SimplifiedFastCIFAR10Net(nn.Module):
    def __init__(self):
        super(SimplifiedFastCIFAR10Net, self).__init__()
        
        # Define channel sizes
        self.prep_channels = 64
        self.layer1_channels = 128
        self.layer2_channels = 256
        self.layer3_channels = 512

        self.main = nn.Sequential(
            # Prep layer
            nn.Conv2d(in_channels=3, out_channels=self.prep_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.prep_channels),
            nn.ReLU(inplace=True),
            
            # Layer 1
            nn.Conv2d(in_channels=self.prep_channels, out_channels=self.layer1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Residual block for Layer 1
            nn.Conv2d(in_channels=self.layer1_channels, out_channels=self.layer1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.layer1_channels, out_channels=self.layer1_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Conv2d(in_channels=self.layer1_channels, out_channels=self.layer2_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer2_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Residual block for Layer 2
            nn.Conv2d(in_channels=self.layer2_channels, out_channels=self.layer2_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer2_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.layer2_channels, out_channels=self.layer2_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer2_channels),
            nn.ReLU(inplace=True),
            
            # Layer 3
            nn.Conv2d(in_channels=self.layer2_channels, out_channels=self.layer3_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer3_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Residual block for Layer 3
            nn.Conv2d(in_channels=self.layer3_channels, out_channels=self.layer3_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer3_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.layer3_channels, out_channels=self.layer3_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.layer3_channels),
            nn.ReLU(inplace=True),
            
            # Final pooling
            nn.MaxPool2d(4),
        )
        
        # Classifier
        self.classifier = nn.Linear(self.layer3_channels, 10, bias=False)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR10 dataset
try:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
except Exception as e:
    print(f"Error loading CIFAR10 dataset: {e}")
    raise

# Initialize the model, loss function, and optimizer
net = SimplifiedFastCIFAR10Net().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Using label smoothing
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

# Initialize the GradScaler for mixed precision training
scaler = GradScaler()

# Training function
def train(epochs):
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Evaluate on the test set
        accuracy = evaluate()
        print(f'Epoch {epoch + 1} Accuracy: {accuracy:.2f}%')

        # Update learning rate
        scheduler.step(accuracy)

        # Save the best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(net.state_dict(), 'best_model.pth')
            
        if accuracy > 93.0:
            print(f"Reached 93% accuracy at epoch {epoch + 1}")
            break

    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds')
    print(f'Best accuracy: {best_acc:.2f}%')

# Evaluation function
def evaluate():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main execution
if __name__ == '__main__':
    try:
        epochs = 100  # Adjust as needed
        train(epochs)
        print("Training and evaluation completed successfully.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise