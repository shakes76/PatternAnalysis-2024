import os
import cv2 as cv


ISC2018_TRUTH_PATH = ".\data-ISC2018\ISIC2018_Task1_Training_GroundTruth_x2"
CLASS_NO = "0" #Only testing for 1 class: skin legions.

"""
create_label(mask)
    Finds the contours from a given mask, uses those contours to find the bounding Rectangle
    and then returns this rectangle in YOLO format
    mask: (numpy array) an array containing greyscale image data.
"""
def create_label(mask):
    mask_height, mask_width, _ = mask.shape
    contours, _ = cv.findContours(mode = cv.RETR_TREE, method = CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv.bountingRect(contours[0])

    #Normalize x, y, w, h because cv2 uses a top left format to store coordinates for bountingRect, yolo stores its coordinates in center format
    #Format details can be found here: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    x_n = (x + w/2) / mask_width
    y_n = (y + h/2) / mask_height
    w_n = w / mask_width
    h_n = h / mask_height
    label = f"{CLASS_NO} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}"
    return label

#Make directories needed for processing if they do not already exist
os.makedirs("./data/train/labels", exist_ok=True)

for file in os.listdir(ISC2018_TRUTH_PATH):
    if file.endswith(".png"):
        mask_path = os.path.join(ISC2018_TRUTH_PATH, file)
        label_path = os.path.join("./data/train/labels", file.replace("_segmentation.png", ".txt"))

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError("Not Found: " + mask_path)
        label = create_label(mask)
        if label:
            with open(file, w) as f:
                f.write(label)
            print("Created label for: " + file)
        else:
            print(file + " is missing contours.")