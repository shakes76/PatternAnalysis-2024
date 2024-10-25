import cv2
import os

# Paths for input data and output labels
HOME_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\ISIC-2017_Training_Part1_GroundTruth'
OUTPUT_PATH = r'C:\Users\raulm\OneDrive\Desktop\Yolo\train\labels'

def ensure_directory(path):
    """
    Ensures that the specified directory exists. If it doesn't it gets created.
    :param path: The path of the directory.
    """
    os.makedirs(path, exist_ok=True)

def read_image(img_path):
    """
    Reads an image from the specified path in grayscale.
    :param img_path: The file path of the image to read.
    :return: The loaded image in grayscale.
    :raises FileNotFoundError: If the image is not found at the specified path.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img

def extract_bounding_box(img):
    """
    Extracts the bounding box from the largest external contour in the image.
    :param img: The input image (grayscale).
    :return: A tuple (x, y, w, h) representing the bounding box, or None if no contour is found.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

def normalise_bbox(x, y, w, h, img_shape):
    """
    Normalises the bounding box coordinates to be relative to the image dimensions.
    :param x: X-coordinate of the top-left corner of the bounding box.
    :param y: Y-coordinate of the top-left corner of the bounding box.
    :param w: Width of the bounding box.
    :param h: Height of the bounding box.
    :param img_shape: Shape of the image as (height, width).
    :return: A tuple (x_center, y_center, w_norm, h_norm) with normalised coordinates.
    """
    height, width = img_shape[:2]
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    w_norm = w / width
    h_norm = h / height
    return x_center, y_center, w_norm, h_norm

def save_label(filename, data):
    """
    Saves the normalized bounding box data to a text file in YOLO format.
    :param filename: The output file path where the label will be saved.
    :param data: A tuple (x_center, y_center, w_norm, h_norm) of normalised coordinates.
    """
    with open(filename, 'w') as f:
        f.write(f"0 {data[0]:.6f} {data[1]:.6f} {data[2]:.6f} {data[3]:.6f}")

def process_masks(input_dir, output_dir):
    """
    Processes all segmentation mask images in the input directory.
    :param input_dir: Directory containing segmentation mask images.
    :param output_dir: Directory where the corresponding labels will be saved.
    """
    ensure_directory(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('_segmentation.png'):
            img_path = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename.replace("_segmentation.png", ".txt"))

            try:
                img = read_image(img_path)
                bbox = extract_bounding_box(img)

                if bbox:
                    normalized_bbox = normalise_bbox(*bbox, img.shape)
                    save_label(output_file, normalized_bbox)
                    print(f"Processed: {filename}")
                else:
                    print(f"No contours found in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    process_masks(HOME_PATH, OUTPUT_PATH)
