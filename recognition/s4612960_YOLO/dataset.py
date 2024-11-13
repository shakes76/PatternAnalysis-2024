import cv2
import os

SRC_DIR = r'/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Validation_Part1_GroundTruth'
DEST_DIR = r'/Users/mariam/Downloads/COMP3710_YOLO/val/labels'

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def load_image_as_grayscale(path):
    grayscale_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if grayscale_img is None:
        raise FileNotFoundError(f"Could not locate image: {path}")
    return grayscale_img

def locate_bounding_box(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(contours[0])

def save_as_yolo_format(filepath, coordinates):
    with open(filepath, 'w') as file:
        file.write(f"0 {coordinates[0]:.6f} {coordinates[1]:.6f} {coordinates[2]:.6f} {coordinates[3]:.6f}")

def process_segmentation_files(source, destination):
    create_directory(destination)
    
    for file in os.listdir(source):
        if file.endswith('_segmentation.png'):
            img_path = os.path.join(source, file)
            label_file = os.path.join(destination, file.replace("_segmentation.png", ".txt"))

            try:
                img = load_image_as_grayscale(img_path)
                bbox = locate_bounding_box(img)

                if bbox:
                    normalized_coords = normalize_coordinates(*bbox, img.shape)
                    save_as_yolo_format(label_file, normalized_coords)
                    print(f"Processed: {file}")
                else:
                    print(f"No bounding box found in {file}")
            except Exception as error:
                print(f"Failed to process {file}: {error}")

if __name__ == "__main__":
    process_segmentation_files(SRC_DIR, DEST_DIR)
