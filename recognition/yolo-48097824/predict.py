import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO 

def model():

    model = YOLO('yolo11n.pt')

    return model

def predict(model, image_path):

    image = cv2.imread(image_path)
    results = model.predict(image_path)
    result_img = results[0].plot()

    return result_img

def display(image):

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":

    image_path = '/mnt/c/Users/jccoc/Downloads/ISIC_DATASET/test/images/ISIC_0012086.jpg'

    model = model()
    predicted_image = predict(model, image_path)
    display(predicted_image)

