import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pywt

# Load the trained model
model = tf.keras.models.load_model("seizure_detection_model.h5")

# Function to extract features using Discrete Wavelet Transform (DWT)
def extract_dwt_features(signals):
    features = []
    for signal in signals:
        signal_features = []
        for channel in signal:
            coeffs = pywt.wavedec(channel, 'db4', level=4)
            coeffs_flattened = np.hstack(coeffs)
            signal_features.append(coeffs_flattened)
        features.append(np.array(signal_features))
    return np.array(features)

# Function to load a signal from a CSV file and reshape it for prediction
def load_signal_from_csv(file_path):
    signal_df = pd.read_csv(file_path)
    signal = signal_df.values
    signal = signal.reshape((1, signal.shape[0], signal.shape[1]))
    return signal

# Function to predict seizure or normal from a CSV file
def predict_seizure(file_path):
    signal = load_signal_from_csv(file_path)
    print("Shape of raw test:", signal.shape)
    signal_f = extract_dwt_features(signal)
    print("Shape of feature test:", signal_f.shape)
    prediction = model.predict(signal_f)
    return "Seizure" if prediction[0][0] > 0.5 else "Normal"

# Example usage
csv_file = "dataset/signal_1_label_0.csv"
prediction = predict_seizure(csv_file)
print(f"The prediction for {csv_file} is: {prediction}")
