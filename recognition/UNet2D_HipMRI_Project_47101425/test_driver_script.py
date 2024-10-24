# Imports
import subprocess
import os

# Define base data directory
data_directory = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/"

# Define image and mask directories for training and validation
train_image_directory = os.path.join(data_directory, "keras_slices_train")
train_mask_directory = os.path.join(data_directory, "keras_slices_seg_train")

validate_image_directory = os.path.join(data_directory, "keras_slices_validate")
validate_mask_directory = os.path.join(data_directory, "keras_slices_seg_validate")


def run_script(script_name, *args):
    """
    Run a specified Python script with given arguments.

    Args:
        script_name (str): The name of the Python script to run.
        *args: Additional arguments to pass to the script.

    Raises:
        subprocess.CalledProcessError: If the script execution fails.
    """
    try:
        print(f"Running {script_name} with arguments: {args}...")
        result = subprocess.run(["python", script_name] + list(args), check=True)
        print(f"{script_name} completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")


if __name__ == "__main__":
    # Run training script with specified arguments
    run_script(
        "train.py", 
        "--base_dir", data_directory, 
        "--image_dir", train_image_directory, 
        "--mask_dir", train_mask_directory
    )

    # Run prediction script with specified arguments
    run_script(
        "predict.py", 
        "--base_dir", data_directory, 
        "--image_dir", validate_image_directory, 
        "--mask_dir", validate_mask_directory
    )
