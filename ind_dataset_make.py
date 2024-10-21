import numpy as np
import pandas as pd
import os

# Load the dataset
file_path = "dataset/eeg-seizure_train.npz"
data = np.load(file_path)
train_signals = data['train_signals']
train_labels = data['train_labels']

# Print the shape of train_signals
print("Shape of train_signals:", train_signals.shape)

# Reduce the dataset size by taking a subset (e.g., 10%)
subset_size = int(0.1 * train_signals.shape[0])
train_signals = train_signals[:subset_size]
train_labels = train_labels[:subset_size]

# Function to save signals to CSV files
def save_signals_to_csv(signals, labels, output_dir):
    for i in range(signals.shape[0]):
        signal_df = pd.DataFrame(signals[i])
        label = labels[i]
        file_name = f"{output_dir}/signal_{i}_label_{label}.csv"
        signal_df.to_csv(file_name, index=False)

# Create an output directory
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Save the signals to CSV files
save_signals_to_csv(train_signals, train_labels, output_dir)

print(f"Signals saved to {output_dir} directory.")
