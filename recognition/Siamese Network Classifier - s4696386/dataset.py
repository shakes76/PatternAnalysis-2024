
from tkinter import filedialog
import os
import matplotlib.pyplot as plt

BENIGN = 0
MALIGNANT = 1

def read_data(file_path_image_folder: str = None, file_path_ground_truth: str = None):
    
    # Ensure we have a directory to take data from
    if file_path_image_folder is None:
        file_path_image_folder = filedialog.askdirectory()
    # Ensure we have a collection of ground truths
    if file_path_ground_truth is None:
        file_path_ground_truth = filedialog.askopenfile().name
    
    # Move to that directory as our current working directory
    os.chdir(file_path_image_folder)
    
    # Create dictionary mapping image names to their file names
    files: dict = {file.removesuffix(".dcm"): file for file in os.listdir()}
    
    # Create dictionary mapping image names to (patient_id, malignant)
    truths: dict = {}
    # Maintain list of malignant & benign images
    malignants: list[str] = []
    benigns: list[str] = []
    # Populate dict and lists
    with open(file_path_ground_truth) as file_ground_truth:
        for i, line in enumerate(file_ground_truth):
            if i == 0:
                continue
            image_name, patient_id,_,_,_,_, malignant, *_ = line.split(",")
            # Assign numerical values to malignance
            malignant = BENIGN if "benign" in malignant else MALIGNANT
            truths[image_name] = patient_id, malignant
            if malignant:
                malignants.append(image_name)
            else:
                benigns.append(image_name)
    
    return files, truths, malignants


# Main function for profiling & debugging
def main():
    import cProfile
    import pstats

    current_directory = os.getcwd()
    with cProfile.Profile() as pr:
        read_data("C:/Users/Kai Graz/Documents/University/2024 Semester 2/COMP3710/Final Project/ISIC_2020_TRAIN", "C:/Users/Kai Graz/Documents/University/2024 Semester 2/COMP3710/Final Project/ISIC_2020_Training_GroundTruth.csv")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    os.chdir(current_directory)
    stats.dump_stats(filename="profile.prof")

# Only run main when running file directly (not during imports)
if __name__ == "__main__":
    main()






















