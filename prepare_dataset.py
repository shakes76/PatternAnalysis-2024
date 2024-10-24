import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def move_dataset():
    # Source directories of data
    input_dir = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
    mask_dir = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'

    # Create directories to move data to
    for dir_path in ['data/images/train', 'data/images/val', 'data/masks/train', 'data/masks/val']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Get list of image files (.jpg)
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            image_files.append(filename)

    # Split the dataset into train and validation sets (80-20 split)
    train_files, val_files = train_test_split(
        image_files,
        test_size=0.2,
        random_state=40
    )

    # Helper used to get mask's file name from the image's name
    def get_mask_filename(image_filename):
        return image_filename.replace('.jpg', '_segmentation.png')

    # Copy training files
    for filename in train_files:
        # Copy input image
        shutil.copy2(
            os.path.join(input_dir, filename),
            os.path.join('data/images/train', filename)
        )
        # Copy mask
        mask_filename = get_mask_filename(filename)
        shutil.copy2(
            os.path.join(mask_dir, mask_filename),
            os.path.join('data/masks/train', mask_filename)
        )

    # Copy validation files
    for filename in val_files:
        # Copy input image
        shutil.copy2(
            os.path.join(input_dir, filename),
            os.path.join('data/images/val', filename)
        )
        # Copy mask
        mask_filename = get_mask_filename(filename)
        shutil.copy2(
            os.path.join(mask_dir, mask_filename),
            os.path.join('data/masks/val', mask_filename)
        )

def generate_labels():
    # Create the directories to store the ground truth label txt files
    os.makedirs('data/labels/train', exist_ok=True)
    os.makedirs('data/labels/val', exist_ok=True)

    # The directories that store the binary mask images and the directories to store the ground truth label txt files
    input_dir_train = './data/masks/train'
    output_dir_train = './data/labels/train'
    input_dir_val = './data/masks/val'
    output_dir_val = './data/labels/val'
    dirs_pairs = [[input_dir_train, output_dir_train], [input_dir_val, output_dir_val]]

    # Create the ground truth label txt files for the training and validation sets to suit the Ultralytics's yolo model
    for dirs_pair in dirs_pairs:
        input_dir = dirs_pair[0]
        output_dir = dirs_pair[1]
        for file_name in os.listdir(input_dir):
            image_path = os.path.join(input_dir, file_name)
            
            # This is used to get the binary mask image in order to retrieve the contours
            # Ultralytics's yolo model requires the mask to be in the format of a polygon
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # find the contours
            height, width = mask.shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Change the contours to polygons
            polygons_list = []
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    polygon = []
                    for point in contour:
                        x, y = point[0]
                        # Normalize the points
                        polygon.append(x / width)
                        polygon.append(y / height)
                    polygons_list.append(polygon)

            # Put the polygons into the txt file
            polyglon_file_name = f"{os.path.splitext(os.path.join(output_dir, file_name))[0]}.txt"

            with open(polyglon_file_name, 'w') as file:
                for polygon in polygons_list:
                    for index, p in enumerate(polygon):
                        if index == len(polygon) - 1:
                            file.write('{}\n'.format(p))
                        elif index == 0:
                            file.write('0 {} '.format(p))
                        else:
                            file.write('{} '.format(p))

                file.close()

def rename_groundtruth():
    # List of directories to process
    directories = ['data/labels/train', 'data/labels/val']

    # Process each directory
    for directory in directories:
        # Get all files in directory
        files = os.listdir(directory)
        
        # Go through each file
        for filename in files:
            if '_segmentation.txt' in filename:
                # Create new filename by replacing '_segmentation' with ''
                new_filename = filename.replace('_segmentation.txt', '.txt')
                
                # Generate full paths to do rename
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                # Rename file
                os.rename(old_path, new_path)

move_dataset()
generate_labels()
rename_groundtruth()