import train
import matplotlib.pyplot as plt
from train import unetModel, dice_metric, trainResults
from dataset import testImages, testSegImages
import numpy as np  

testPredictedSeg = unetModel.predict(testImages)
print(np.unique(testPredictedSeg)) 

#Function to find the dice score of each set of actual segments and predicted
def calculate_dice_scores(y_true, y_pred):
    dice_scores = []
    for i in range(len(y_true)):
        y_pred_squeezed = np.squeeze(y_pred[i])  
        score = dice_metric(y_pred_squeezed, y_true[i]).numpy()  
        dice_scores.append(score)
    return dice_scores

dice_scores = calculate_dice_scores(testSegImages, testPredictedSeg)
dice_scores = np.array(dice_scores)

#Print the actual image, actual segment and predicted segment 
fig, pos = plt.subplots(5, 3, figsize=(15, 25))
for i in range(5):
    # Display original image
    pos[i, 0].imshow(testImages[i].squeeze())
    pos[i, 0].set_title(f'Original image {i+1}')
    pos[i, 0].axis('off')

    # Display actual segmentation 
    pos[i, 1].imshow(testSegImages[i].squeeze())
    pos[i, 1].set_title(f'Actual segmentation {i+1}')
    pos[i, 1].axis('off')

    # Display predicted segmentation 
    pos[i, 2].imshow(testPredictedSeg[i].squeeze())
    pos[i, 2].set_title(f'Predicted segmentation {i+1}')
    pos[i, 2].axis('off')
plt.tight_layout()
plt.show()

#print the dice scores for each image and the distribution
plt.figure(figsize=(12, 6))
plt.plot(dice_scores, marker='o', linestyle='None', color='b')
plt.title('Dice Scores for Each Test Image')
plt.xlabel('Test Image Index')
plt.ylabel('Dice Score')
plt.ylim(0, 1) 
plt.yticks(np.linspace(0, 1, num=11))  
plt.grid()
plt.axhline(y=np.mean(dice_scores), color='r', linestyle='--', label='Mean Dice Score')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(dice_scores, bins=10, color='c', edgecolor='black', alpha=0.7)
plt.title('Distribution of Dice Scores')
plt.xlabel('Dice Score')
plt.ylabel('Frequency')
plt.xlim(0, 1) 
plt.grid()
plt.show()

# Plotting Loss and Dice Coefficient
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(trainResults.history['loss'], label='Training Loss')
plt.plot(trainResults.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot Dice Coefficient
plt.subplot(1, 2, 2)
plt.plot(trainResults.history['dice_metric'], label='Training Dice Coefficient')
plt.plot(trainResults.history['val_dice_metric'], label='Validation Dice Coefficient')
plt.title('Dice Coefficient Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
