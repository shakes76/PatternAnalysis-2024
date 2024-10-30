import os
import cv2 as cv


ISC2018_TRUTH_TRAIN = "./data-ISC2018/ISIC2018_Task1_Training_GroundTruth_x2"
ISC2018_TRUTH_VALIDATE = "./data-ISC2018/ISIC2018_Task1_Validation_GroundTruth"
ISC2018_TRUTH_TEST = "./data-ISC2018/ISIC2018_Task1_Test_GroundTruth"
OUTPUT_TRAIN = "./data/labels/train"
OUTPUT_VALIDATE = "./data/labels/validate"
OUTPUT_TEST = "./data/labels/test"
CLASS_NO = "0" #Only testing for 1 class: skin legions.

"""
create_label(mask)
    Finds the contours from a given mask, uses those contours to find the bounding Rectangle
    and then returns this rectangle in YOLO format
    mask: (numpy array) an array containing greyscale image data.
"""
def create_label(mask):
    mask_height, mask_width = mask.shape
    contours, _ = cv.findContours(mask, mode = cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv.boundingRect(contours[0])

    #Normalize x, y, w, h because cv2 uses a top left format to store coordinates for bountingRect, yolo stores its coordinates in center format
    #Format details can be found here: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    x_n = (x + w/2) / mask_width
    y_n = (y + h/2) / mask_height
    w_n = w / mask_width
    h_n = h / mask_height
    label = f"{CLASS_NO} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}"
    return label

"""
mask2label(input_path, output_path)
    Converts all ground truth masks within a folder to YOLO format labels
    and then returns this rectangle in YOLO format
    input_path: input location
    output_path: output location
"""
def mask2label(input_path, output_path):
    for file in os.listdir(input_path):
        if file.endswith(".png"):
            mask_path = os.path.join(input_path, file)
            label_path = os.path.join(output_path, file.replace("_segmentation.png", ".txt"))

            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError("Not Found: " + mask_path)
            label = create_label(mask)
            if label:
                with open(label_path, "w") as f:
                    f.write(label)
                #hidden because printing was actually slowing the process down
                #print("Created label for: " + file)
            else:
                print(file + " is missing contours.")


#Make directories needed for processing if they do not already exist
os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VALIDATE, exist_ok=True)
mask2label(ISC2018_TRUTH_TRAIN, OUTPUT_TRAIN)
mask2label(ISC2018_TRUTH_VALIDATE, OUTPUT_VALIDATE)
mask2label(ISC2018_TRUTH_TEST, OUTPUT_TEST)
