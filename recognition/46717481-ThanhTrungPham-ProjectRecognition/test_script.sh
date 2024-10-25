#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Preprocess the dataset
python3 dataset.py

# Run the training
python3 train.py

# Run the test to see a sample of the prediction
python3 predict.py