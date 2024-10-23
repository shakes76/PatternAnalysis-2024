import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from modules import SiameseNetwork

class PredictData:
    def __init__(self, test_data, path, show_images=False):
        '''
        Initialize the PredictData class.

        Args:
            test_data (iterable): Test dataset to evaluate the model.
            path (str): Path to the project directory where outputs and results will be saved.
            show_images (bool): Whether to show images during the prediction process.
        '''
        self.test_data = test_data
        self.path = path
        self.show_images = show_images

        # Results path where final plots are saved
        self.results_path = os.path.join(self.path, "results")

        # The path to the model's checkpoint - where weights are saved
        # The checkpoint is kind of like a list of different checkpoints, hence why we need to index it with 'backbone'
        self.checkpoint_path = os.path.join(self.path, "outputs", "best.pth")

        # Set device to CUDA if a CUDA device is available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # The inverse transform to reverse the normalization process, so we can visualize the data
        self.inv_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

        self.criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss

        # Load the model checkpoint and set up the model
        self.checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        self.model = SiameseNetwork(backbone=self.checkpoint['backbone'])
        self.model.to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()  # Set the model to evaluation mode

        # Initialize tracking variables for losses and accuracy
        self.losses = []
        self.correct = 0
        self.total = 0
        self.print_frequency = 100  # Frequency at which results are printed

    def predict(self):
        '''
        Predict the similarity between pairs of images in the test dataset and evaluate model performance.
        '''
        for i, ((img1, img2), target, (class1, class2)) in enumerate(self.test_data):
            # Move the images and target to the appropriate device (GPU or CPU)
            img1, img2, target = map(lambda x: x.to(self.device), [img1, img2, target])
            class1 = class1[0]
            class2 = class2[0]

            # Pass images through the model to get similarity prediction
            similarity = self.model(img1, img2)
            # Calculate loss using Binary Cross-Entropy
            loss = self.criterion(similarity, target)

            # Store loss and calculate correct predictions
            self.losses.append(loss.item())
            self.correct += torch.count_nonzero(target == (similarity > 0.5)).item()
            self.total += len(target)

            # Print progress at regular intervals
            if (i % self.print_frequency) == 0:
                print("[{} / {}]".format(i, len(self.test_data)))

            # Plot and save images being compared if show_images is enabled
            fig = plt.figure("class1={}	class2={}".format(class1, class2), figsize=(4, 2))
            plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(class1, similarity[0][0].item(), class2))

            if (i % self.print_frequency) == 0:
                # Apply inverse transform (de-normalization) to retrieve original images.
                img1 = self.inv_transform(img1).cpu().numpy()[0]
                img2 = self.inv_transform(img2).cpu().numpy()[0]

                # Show first image
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(np.transpose(img1, (1, 2, 0)))
                plt.axis("off")

                # Show second image
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(np.transpose(img2, (1, 2, 0)))
                plt.axis("off")

                # Save the plot to the results directory
                save_path = os.path.join(self.results_path, f'prediction_results_{i}.png')
                fig.savefig(save_path, format='png')

        # Print the final validation loss and accuracy
        print("Validation: Loss={:.2f}	 Accuracy={:.2f}	".format(sum(self.losses) / len(self.losses),
                                                                   self.correct / self.total))
