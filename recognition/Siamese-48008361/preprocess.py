import os
import shutil
import pandas as pd
from tqdm import tqdm

def preprocess_dataset(csv_file, img_dir, output_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create output directories
    benign_dir = os.path.join(output_dir, 'benign')
    malignant_dir = os.path.join(output_dir, 'malignant')
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_name = row['image_name'] + '.jpg'
        src_path = os.path.join(img_dir, img_name)
        
        if row['target'] == 0:  # Benign
            dst_path = os.path.join(benign_dir, img_name)
        else:  # Malignant
            dst_path = os.path.join(malignant_dir, img_name)
        
        shutil.copy(src_path, dst_path)

    print(f"Preprocessing complete. Images organized in {output_dir}")

if __name__ == "__main__":
    csv_file = 'ISIC_2020_Training_GroundTruth_v2.csv'
    img_dir = 'data/ISIC_2020_Training_JPEG/train/'
    output_dir = 'preprocessed_data'
    
    preprocess_dataset(csv_file, img_dir, output_dir)