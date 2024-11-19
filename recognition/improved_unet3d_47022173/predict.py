"""
This file contains example usage of the model (training and testing) with code to create 
visualisation. See READEME.md for actual visualisations and results.

Abdullah Badat (47022173), abdullahbadat27@gmail.com
"""
import ast
from train import *
import matplotlib.pyplot as plt

# Training and validation
train(
    "train",
    "./data/semantic_MRs_anon/",
    "./data/semantic_labels_anon/",
    0.0001,
    0.0001,
    10,
    0.1,
    50,
    2,
    "saves"
)

# Read out file generated during training
training_loss = {}
validation_loss = {}
validation_dice_scores = {}
with open("out_train.txt", "r") as f:
    epoch = 0
    for line in f:

        if epoch == 50:
            break

        line = line.strip()
        if line[0] == "[":
            validation_dice_scores[epoch] = ast.literal_eval(line)

        elif line[0] == "E":
            epoch += 1
            training_loss[epoch] = float(line.split()[4])
        elif line[0] == "V":
            validation_loss[epoch] = 1 - float(line.split()[2])

# Plot training loss
plt.title("Training and Validation Loss")
plt.plot(list(training_loss.keys()), list(training_loss.values()), label="Training Loss")
plt.plot(list(validation_loss.keys()), list(validation_loss.values()), label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss_plot.png')

# Plot the dice scores as separate lines on a line graph
plt.title("Validation Dice Scores")
for i in range(6):
    plt.plot(list(validation_dice_scores.keys()), [x[i] for x in validation_dice_scores.values()],
              label=f"Class {i}")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.legend()
plt.show()
plt.savefig('dice_scores.png')

# Running testing with trained model
test(
    "saves/model.pth",
    "./data/semantic_MRs_anon/",
    "./data/semantic_labels_anon/",
    "saves"
)

# Testing Dice scores
training_loss = {}
validation_loss = {}
validation_dice_scores = {}
with open("out_test.txt", "r") as f:
    epoch = 0
    for line in f:
        line = line.strip()
        if line[0] == "[":
            validation_dice_scores[epoch] = ast.literal_eval(line)

# Plot the dice scores as a bar graph
x = validation_dice_scores.keys()
y = validation_dice_scores.values()
plt.bar(list(x), list(y))
plt.title("Test Set Dice scores")
plt.axhline(y=0.7, color='r', linestyle='-')
plt.xlabel("Class")
plt.ylabel("Dice Score")
plt.savefig('test_dice_scores.png')