#!/bin/bash
#SBATCH --time=0-00:30:00           # Time limit: 30 minutes
#SBATCH --nodes=1                   # Run on single node
#SBATCH --ntasks-per-node=1         # Single task per node
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --partition=gpu            # GPU partition
#SBATCH --job-name="yolov8-victor"          # Job name
#SBATCH --mail-user=phamtrung0633@email.com  # Email address for notifications
#SBATCH --mail-type=BEGIN          # Email at start
#SBATCH --mail-type=END            # Email at completion
#SBATCH --mail-type=FAIL           # Email on failure
#SBATCH --output=yolo_train_%j.out  # Output file

# Load required modules
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Set Python path

# Run the training script
python train_model.py

echo "End Time: $(date)"