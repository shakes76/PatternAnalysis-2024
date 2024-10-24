import os
import cv2 

def normalise_point(point, img_shape):

    x_normalised = int(point[0][0]) / img_shape[1]
    y_normalised = int(point[0][1]) / img_shape[0]

    return str(x_normalised) + " " + str(y_normalised)

def polygon_from_mask(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    polygons = []
    img_shape = img.shape 

    for contour in contours:
        for point in contour:
            polygons.append(normalise_point(point, img_shape))

    return " ".join(polygons)

def process_masks(data_path, output_location):

    for filename in os.listdir(data_path):
        if filename.endswith('.png'):
            img_path = os.path.join(data_path, filename)
            annotation_file = filename.replace("_segmentation.png", ".txt")
            annotation_path = os.path.join(output_location, annotation_file)

            with open(annotation_path, 'w') as f:
                f.write("0 ")  
                f.write(polygon_from_mask(img_path))

def main():
    
    data_path = "/mnt/c/Users/jccoc/Downloads/ISIC_DATASET/test/masks"
    output_location = "/mnt/c/Users/jccoc/Downloads/ISIC_DATASET/test/labels"
    process_masks(data_path, output_location)

if __name__ == "__main__":
    main()