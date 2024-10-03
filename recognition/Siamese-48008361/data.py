import numpy as np
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    # Load the image using PIL
    image = Image.open(image_path)

    # Resize the image to the desired target size
    image = image.resize(target_size)

    # Convert to NumPy array and normalize pixel values to [0, 1]
    image_array = np.array(image) / 255.0

    return image_array

def testImage():
        
    image_dir = 'data/ISIC_2020_Training_JPEG/train/'
    image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]

    
    preprocessed_image = load_and_preprocess_image(image_files[0])
    print(preprocessed_image.shape) 
    print(len(image_files))

    # Display the first preprocessed image
    

    plt.imshow(preprocessed_image)
    plt.title("Preprocessed Image")
    plt.axis('off')
    plt.show()

