"""
Used to run inference on either the test dataset or a specified image.
Returns pertinent result metrics alongside inference outcomes.

Usage example: python predict.py -p image -img Data/Testing/images/image0.png -w yolo/best_tuned.pt

@author Ewan Trafford
"""

import modules
import dataset
import argparse


def IoUTotalUpdate(label_bbox, inference_bbox):
    """
    Calculates updates made to the running total intersection over union. 
    Compares the corner coordinates of the target bounding box w/ the predicted bounding box.

    Args:
        label_bbox (array): Four normalised coordinates of target bounding box.
        inference_bbox (array): Four normalised coordinates of predicted bounding box.
    """

    x1_min, y1_min, x1_max, y1_max = bbox_to_corners(label_bbox)
    if inference_bbox:
        x2_min, y2_min, x2_max, y2_max = bbox_to_corners(inference_bbox[0])
    else: # if no lesion is found
        return 0
    
    # Calculate intersection coordinates
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    # Calculate intersection area
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    intersection_area = inter_width * inter_height
    
    # Calculate area of each bounding box
    area_bbox1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_bbox2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area
    
    # Compute IoU
    IoUAdded = intersection_area / union_area if union_area > 0 else 0
    return IoUAdded

def LabelReader(index):
    """
    Used to extract the bounding box coordinates for the image at the index passed in.

    Args:
        Index (int): The position of the image of concern in its dataset.
    """
    with open('Data/Testing/labels/image'+str(index)+'.txt', 'r') as file:
        # Read the contents of the file
        line = file.readline().strip()  # Read the first line and remove any leading/trailing whitespace
        # Convert the numbers in the line to a list of floats (or integers if appropriate)
        numbers = [float(num) for num in line.split()]  # Split the line by spaces and convert to floats
        del numbers[0]
    return numbers

def bbox_to_corners(bbox):
    """
    Converts the standard formula of centreX, centreY, width, height all normalised into the four corner coordinates.

    Args:
        bbox (array): An array with four elements, entries representing the standard values above.
    """
    # Convert (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
    center_x, center_y, width, height = bbox
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    return x_min, y_min, x_max, y_max

def inference_on_testset(model):
    """
    Runs inference on the test dataset, calculating the average confidence score and intersection over union once finished

    Args:
        model (YOLO): A specified model that has been trained on weights passed into it.
    """
    testset = dataset.test_dataset
    prediction_array = []
    IoUTotal = 0
    confTotal = 0
    index = 0

    for i in range(0, testset.__len__()):
        if (i % 50) == 49:  # Broken into segments to prevent RAM allocation issues
            print("Running inference on 50 images.")
            results = model(prediction_array, max_det = 1) # run batched inference on the test dataset
            index = 0
            for result in results:
                IoUTotal += IoUTotalUpdate(LabelReader(index), result.boxes.xywhn.tolist())
                if result.boxes.conf.size(0) > 0:
                    confTotal += result.boxes.conf.item()
                if index == 1:
                    result.show()  # Display the results of running inference on every 50th image 
                index += 1
            prediction_array = []

        else:
            prediction_array.append(testset.get_image_path(i))

    print("Average IoU across all predicted images is: " + str(IoUTotal / index))
    print("Average confidence score across all predicted images is: " + str(confTotal / index))

def inference_on_image(image_path, model):
    """
    Runs inference on a single image using YOLOv11's predict mode.
    Saves and displays results

    Args:
        image_path: The relative or absolute path to the image that inference is run on. 
        model (YOLO): A specified model that has been trained on weights passed into it.
    """
    model.predict(image_path, save = True, show = True)


def main():
    parser = argparse.ArgumentParser(description='Run inference on a particular image or the test set.')
    parser.add_argument('-p', type=str, required=True, choices=['testset', 'image'], 
                        help='Specify "testset" or "image".')
    parser.add_argument('-img', type=str, help='Path to the image (required if -p is "image").')
    parser.add_argument('-w', type=str, required=True, 
                        help='Path to the model weights file.')

    args = parser.parse_args()

    model = modules.YOLOv11(args.w)

    # Process the command line argument
    if args.p == "testset":
        print("Processing the test set...")
        # Call your function to handle test set processing
        inference_on_testset(model)
    elif args.p == "image":
        if args.img:
            image_path = args.img
            print(f"Processing image at path: {image_path}")
            # Call your function to handle image processing
            inference_on_image(image_path, model)
        else:
            print("Error: If using 'image', you must provide an image path with -img.")

    
if __name__ == "__main__":
    main()
   