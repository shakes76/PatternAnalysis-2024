import train
import matplotlib.pyplot as plt
from train import trainingLoss, trainingAccuracy, trainingValLoss, trainingValAccuracy, trainingDiceScore, trainingValDiceScore, testLoss, testAccuracy, testDiceScore


# Printing the test stats 
print(testLoss)
print(testAccuracy)
print(testDiceScore)

epochs = 50
# Plotting each of the model training results 
plt.figure(figsize=(10, 5))
plt.plot(epochs, trainingLoss, label='Training Loss', color='blue')
plt.plot(epochs, trainingValLoss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, trainingAccuracy, label='Training Accuracy', color='blue')
plt.plot(epochs, trainingValAccuracy, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, trainingDiceScore, label='Training Dice Score', color='blue')
plt.plot(epochs, trainingValDiceScore, label='Validation Dice Score', color='red')
plt.title('Training and Validation Dice Score')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()
plt.grid()
plt.show()
