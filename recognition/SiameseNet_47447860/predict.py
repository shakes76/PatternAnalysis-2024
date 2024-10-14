import os
import argparse

#import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from modules import SiameseNetwork
from dataset_3 import Dataset

class PredictData:
    def __init__(self, test_data, show_images=False):
        self.test_data = test_data
        self.show_images = show_images

        # results path where our final plots go
        self.results_path = r"C:\Users\sebas\project\results"
        #self.results_path = "~/project/results/"

        # The path to the model's checkpoint - where weights are saved
        # The checkpoint is kind of like a list of different checkpoints, hence why we need to index it with 'backbone'
        self.checkpoint_path = r"C:\Users\sebas\project\outputs\best.pth"
        #self.checkpoint_path = "~/project/outputs/best.pth"

        # Set device to CUDA if a CUDA device is available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # The inverse transform to reverse the normalisation process, so we can visualise the data
        self.inv_transform = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                 std=[1., 1., 1.]),
                                            ])

        self.criterion = torch.nn.BCELoss()

        self.checkpoint = torch.load(self.checkpoint_path)
        self.model = SiameseNetwork(backbone=self.checkpoint['backbone'])
        self.model.to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        self.losses = []
        self.correct = 0
        self.total = 0
        self.print_image_frequency = 1000

    def predict(self):
        # Will have to change the format of the data to fit this for-loop structure
        for i, ((img1, img2), target, (class1, class2)) in enumerate(self.test_data):
            print("[{} / {}]".format(i, len(self.test_data)))

            img1, img2, target = map(lambda x: x.to(self.device), [img1, img2, target])
            class1 = class1[0]
            class2 = class2[0]

            similarity = self.model(img1, img2)
            loss = self.criterion(similarity, target)

            self.losses.append(loss.item())
            self.correct += torch.count_nonzero(target == (similarity > 0.5)).item()
            self.total += len(target)

            fig = plt.figure("class1={}\tclass2={}".format(class1, class2), figsize=(4, 2))
            plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(class1, similarity[0][0].item(), class2))

            # Show the images being compared if we want -> only show some with the frequency parameter
            if self.show_images and (i % self.print_image_frequency == 0):
                # Apply inverse transform (denormalization) on the images to retrieve original images.
                img1 = self.inv_transform(img1).cpu().numpy()[0]
                img2 = self.inv_transform(img2).cpu().numpy()[0]
                # show first image
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(img1[0], cmap=plt.cm.gray)
                plt.axis("off")

                # show the second image
                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(img2[0], cmap=plt.cm.gray)
                plt.axis("off")

            # save the plot
            save_path = os.path.join(self.results_path, 'prediction_results.png')
            fig.savefig(save_path, format='png')

        print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(self.losses) / len(self.losses),
                                                                   self.correct / self.total))
