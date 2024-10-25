# Source code for training, validating, testing and saving the UNet3D Model.
from modules import *
from dataset import *
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import glob
torch.cuda.empty_cache()


cust_dataset = MyCustomDataset()
train, val, test = torch.utils.data.random_split(cust_dataset, [0.2, 0.1, 0.7])
train = DataLoader(train)
val = DataLoader(val)
test = DataLoader(test)
model = ImprovedUNet3D(in_channels=1, out_channels=6).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


criterion = DSC()

learn_rate = 0.1
# Alternates minimum and maximum learning rate
l_sched_1 = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr = learn_rate, step_size_up=15, step_size_down=15, mode='triangular', verbose= False)
# Linearly reduces learning rate
l_sched_2 = optim.lr_scheduler.LinearLR(optimizer , start_factor=0.005/learn_rate, end_factor=0.005/5, verbose=False)
# Sequentially applies sched 1 and 2 
scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[l_sched_1, l_sched_2], milestones=[30])


# No. of Epochs (Number of times all training data is run)
n_epochs = 1
train_losses = []
print(">> Training <<")
start = time.time()
for epoch in range(n_epochs):
    epstart = time.time()
    # Set training mode
    model.train()
    running_loss = 0.0
    print(len(train))
    for i, (inputs, labels) in enumerate(train):

        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    # Set evaluation mode for validation
    train_loss = running_loss / len(train)
    train_losses.append(train_loss)
    model.eval()
    scores = []
    with torch.no_grad():
        for i, (val_inputs, val_labels, ) in enumerate(val):
            val_inputs = val_inputs.cuda()
            val_labels = val_labels.cuda()
            output = model(val_inputs)
            # Calculating Dice score for each pass
            score = 1 - criterion(output, val_labels)
            scores.append(score)
    # Average validation score for all batches
    mean_score = sum(scores) / len(scores)
    epend = time.time()
    print(f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {loss.item()}, Validation DSC: {mean_score}")
    print(f"Epoch {epoch + 1}/{n_epochs} took {epend - epstart} seconds")

end = time.time()
total_time = end - start
print(f"Training and Validation took {total_time} seconds or {total_time/60} minutes.")
