# Example usage of the trained UNet3D model.
import torch
import matplotlib.pyplot as plt

from skimage.util import montage
import numpy as np


#dataset = MyCustomDataset()
#_, test = torch.utils.data.random_split(cust_dataset, [0.9, 0.1])
#test_loader = DataLoader(test)
#model = ImprovedUNet3D(in_channels=1, out_channels=6).cuda()
#model.load_state_dict(torch.load('./new_folder/model.pth'))
def predict(model, test_loader):
    model.eval()
    with torch.no_grad():
        for test_images, test_masks in test_loader:
            # moves test_images to be on gpu
            vis = Visulizer()
            vis.visualize(test_images, test_masks)
            
            test_images = test_images.cuda()
            test_outputs = model(test_images)
            predicted_masks = torch.argmax(test_outputs, dim=1)

            plt.figure(figsize=(10, 5))
            # The base input image
            plt.subplot(1, 3, 1)
            plt.imshow(test_images[0].cpu().squeeze(), cmap='gray')
            plt.title("Input Image")

            # The segmentation mask
            plt.subplot(1, 3, 2)
            plt.imshow(test_masks[0].cpu().argmax(dim=0).squeeze(), cmap='gray')
            plt.title("Segmentation Mask")

            # The predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_masks[0].cpu().squeeze(), cmap='gray')
            plt.title("Predicted Mask")

            # Save image
            output_path = "output.png"
            plt.savefig(output_path)
            plt.show()  
            break
            

class Visulizer:
    def montage_nd(self, image):
        if len(image.shape)>3:
            return montage(np.stack([self.montage_nd(img) for img in image],0))
        elif len(image.shape)==3:
            return montage(image)
        else:
            print('Input less than 3d image, returning original')
            return image

    def visualize(self, image, mask):
        fig, axs = plt.subplots(1, 2, figsize = (20, 15 * 2))
        axs[0].imshow(self.montage_nd(image[...,0]), cmap = 'bone')
        axs[1].imshow(self.montage_nd(mask[...,0]), cmap = 'bone')
        plt.show()
        
