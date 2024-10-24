import cv2
import os

# Paths for input data and output labels
HOME_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\ISIC-2017_Test_v2_Part1_GroundTruth'
OUTPUT_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\test\labels'

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def process_masks(input_dir, output_dir):
    ensure_directory(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('_segmentation.png'):
            img_path = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename.replace("_segmentation.png", ".txt"))
            print("Found")
           
if __name__ == "__main__":
    process_masks(HOME_PATH, OUTPUT_PATH)
