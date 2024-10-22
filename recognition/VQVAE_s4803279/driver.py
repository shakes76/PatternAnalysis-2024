"""
Author: Farhaan Rashid

Student Number: s4803279

Run this file to execute the training and the testing
"""
import os
from train import main
from predict import main_test


def run_train():
    current_location = os.getcwd()
    train_data_dir = os.path.join(current_location, 'recognition', 'VQVAE_s4803279', 'HipMRI_study_keras_slices_data')
    model_output_dir = 'trained_vqvae2_model'
    main(train_data_dir, model_output_dir)


def run_test():
    current_location = os.getcwd()
    model_path = os.path.join(current_location, 'trained_vqvae2_model', 'vqvae2_epoch_final.pth')
    test_data_dir = os.path.join(current_location, 'recognition', 'VQVAE_s4803279', 'HipMRI_study_keras_slices_data', 'keras_slices_test')
    output_dir = 'test_model'
    main_test(model_path, test_data_dir, output_dir)


if __name__ == "__main__":
    run_train()
    run_train()
