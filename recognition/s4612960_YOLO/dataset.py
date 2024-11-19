import cv2
import os

SRC_DIR = r'/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Validation_Part1_GroundTruth'
DEST_DIR = r'/Users/mariam/Downloads/COMP3710_YOLO/val/labels'

def create_directory(path):
    """
    Creates the specified directory if it does not exist.
    :param path: Directory path to check/create.
    """
    os.makedirs(path, exist_ok=True)

def load_image_as_grayscale(path):
    """
    Loads an image from the specified path in grayscale.
    :param path: Path to the image file.
    :return: Grayscale image array.
    :raises FileNotFoundError: If the image does not exist at the specified path.
    """
    grayscale_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if grayscale_img is None:
        raise FileNotFoundError(f"Could not locate image: {path}")
    return grayscale_img

def locate_bounding_box(image):
    """
    Identifies the bounding box of the largest contour in the image.
    :param image: Input grayscale image.
    :return: (x, y, w, h) representing the bounding box, or None if no contours are found.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boundingRect(contours[0])

def save_as_yolo_format(filepath, coordinates):
    """
    Writes normalized bounding box coordinates in YOLO format to a text file.
    :param filepath: Path for the output text file.
    :param coordinates: Normalized bounding box (center_x, center_y, norm_width, norm_height).
    """
    with open(filepath, 'w') as file:
        file.write(f"0 {coordinates[0]:.6f} {coordinates[1]:.6f} {coordinates[2]:.6f} {coordinates[3]:.6f}")


def normalize_coordinates(x, y, width, height, image_size):
    """
    Normalizes bounding box coordinates relative to image dimensions.
    :param x: X-coordinate of the bounding box's top-left corner.
    :param y: Y-coordinate of the bounding box's top-left corner.
    :param width: Width of the bounding box.
    :param height: Height of the bounding box.
    :param image_size: (height, width) of the image.
    :return: (center_x, center_y, norm_width, norm_height) normalized coordinates.
    """
    img_height, img_width = image_size[:2]
    center_x = (x + width / 2) / img_width
    center_y = (y + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return center_x, center_y, norm_width, norm_height


def process_segmentation_files(source, destination):
    """
    Converts each segmentation mask in the source directory to YOLO-format labels and saves them.
    :param source: Path to the directory containing segmentation mask images.
    :param destination: Directory where YOLO format labels will be saved.
    """
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
