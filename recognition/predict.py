import train
import matplotlib.pyplot as plt
from train import unetModel
from dataset import testImages, testSegImages

testPredictedSeg = unetModel.predict(testImages)

fig, pos = plt.subplots(5, 3, figsize=(15, 25))
for i in range(5):
    # Display original image
    pos[i, 0].imshow(testImages[i].squeeze(), cmap='gray')
    pos[i, 0].set_title(f'Original image {i+1}')
    pos[i, 0].axis('off')

    # Display actual segmentation 
    pos[i, 1].imshow(testSegImages[i].squeeze(), cmap='gray')
    pos[i, 1].set_title(f'Actual segmentation {i+1}')
    pos[i, 1].axis('off')

    # Display predicted segmentation 
    pos[i, 2].imshow(testPredictedSeg[i].squeeze(), cmap='gray')
    pos[i, 2].set_title(f'Predicted segmentation {i+1}')
    pos[i, 2].axis('off')

plt.tight_layout()
plt.show()