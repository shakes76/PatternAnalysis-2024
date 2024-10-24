import cv2
import os

# Paths for input data and output labels
HOME_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\ISIC-2017_Training_Part1_GroundTruth'
OUTPUT_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\train\labels'

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)

def read_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img

def extract_bounding_box(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No contour found
    
def process_masks(input_dir, output_dir):
    ensure_directory(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('_segmentation.png'):
            img_path = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename.replace("_segmentation.png", ".txt"))
            
            try:
                img = read_image(img_path)
                bbox = extract_bounding_box(img)
                print(bbox)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
           
if __name__ == "__main__":
    process_masks(HOME_PATH, OUTPUT_PATH)
