from dataset import GFNetDataloader
from modules import GFNet
import torch.optim.adamw
import torch
import time
from torch.optim.lr_scheduler import OneCycleLR
import csv
import argparse

parser = argparse.ArgumentParser(description='Optional argument example')

# Define an optional argument, e.g., '--myarg'
parser.add_argument('--mpath', type=str, help='Model weights to load')
parser.add_argument('--tag', type=str, help='tag to label the models')

# Parse the arguments
args = parser.parse_args()

loaded_model = args.mpath
if not args.tag:
    print('No tag supplied')
    exit(3)
tag = args.tag

# Setting up Coda
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

# hyper-parameters
learning_rate = 0.0008
weight_decay = 0.0001
dropout = 0.0
drop_path = 0.1

batches = 32
patch_size = 64
embed_dim = 192
depth = 12
ff_ratio = 3
epochs = 50

## Load data
gfDataloader = GFNetDataloader(batches)
gfDataloader.load()
training, test, validation = gfDataloader.get_data()
image_size = gfDataloader.get_meta()['img_size']
if not training or not test:
    print("Problem loading data, please check dataset is commpatable with dataloader including all hyprparameters")
    exit(1)


# Image Info
channels = 1
num_classes = 2
img_shape = (channels, image_size, image_size)


model = GFNet(img_size=image_size, patch_size=patch_size, in_chans=channels, num_classes=num_classes, embed_dim=embed_dim, depth=depth, ff_ratio=ff_ratio, dropout=dropout, drop_path_rate=drop_path)
model.to(device)
model.train()
if loaded_model:
    model.load_state_dict(torch.load(loaded_model, weights_only=True))

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = OneCycleLR(optimizer,max_lr=learning_rate, steps_per_epoch=len(training), epochs=epochs)

training_losses = []
test_losses = []
test_accuracy = []


# TODO: Move this to predict
def evaluate_model(loader):
    # Test the model
    print("==Testing====================")
    start_eval = time.time() #time generation
    model.eval() 

    all_preds = []  # To store all predictions
    all_labels = []  # To store true labels

    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = 0 
        count = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            count += 1
            correct += (predicted == labels).sum().item()
            avg_loss += criterion(outputs, labels) 

            # Collect predictions and true labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        test_losses.append(float(avg_loss / count))

        accuracy = (100 * correct / total)
        test_accuracy.append(accuracy)
        print('Test Accuracy: {:.2f} % | Average Loss: {:.4f}'.format(accuracy, avg_loss / count)) 

    end = time.time()
    elapsed = end - start_eval
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    model.train()
    print("=============================")
    return accuracy, (avg_loss / count)

def train_model():
    ## Start training loop
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training, 0):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            train_loss = criterion(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            training_losses.append(train_loss.item())
        
            if i % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}" .format(epoch+1, epochs, i+1, len(training), train_loss.item()))

        result, test_loss = evaluate_model(test) 
        if epoch % 10 == 0:
            result, test_loss = evaluate_model(validation) 
            torch.save(model.state_dict(), 'Checkpoint-GFNET-e{}-{}-{}.pth'.format(epoch, round(result, 4), tag))
            with open('losses-{}.csv'.format(tag), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(training_losses)
            with open('test_losses-{}.csv'.format(tag), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(test_losses)
            with open('test_accuracy-{}.csv'.format(tag), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(test_accuracy)

            if abs(test_loss - train_loss.item()) > 0.35:
                print('=======OVERFITTED-TERMINATED======')
                exit(2)

        scheduler.step() 
    result, final_accuracy = evaluate_model(validation) 
    torch.save(model.state_dict(), 'FINAL_GFNET-{}-{}.pth'.format(round(result, 4), tag))
    with open('losses-{}.csv'.format(tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(training_losses)
    with open('test_losses-{}.csv'.format(tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_losses)
    with open('test_accuracy-{}.csv'.format(tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_accuracy)
    with open('{}-acc.out'.format(tag), 'w', newline='') as f:
        f.write('{} -- Final Accuracy {}\n'.format(tag, final_accuracy))



print("==Training====================")
start = time.time() #time generation
train_model()
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
