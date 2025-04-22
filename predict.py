import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
import pywt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import threading
from playsound import playsound

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
    signal_f = extract_dwt_features(signal)
    prediction = model.predict(signal_f)
    return "Seizure" if prediction[0][0] > 0.5 else "Normal"

# Directory containing the CSV files
directory = "dataset/"
csv_files = [os.path.join(directory, file_name) for file_name in os.listdir(directory) if file_name.endswith(".csv")]

# Initialize the plot
fig, ax = plt.subplots()
lines = []

# Variable to store the previous prediction
previous_prediction = None

# Function to play audio in a separate thread
def play_audio(file):
    threading.Thread(target=playsound, args=(file,)).start()

# Function to update the plot
def update(frame):
    global previous_prediction
    #file_path = csv_files[frame % len(csv_files)]
    file_path = random.choice(csv_files)
    signal = load_signal_from_csv(file_path)
    prediction = predict_seizure(file_path)
    ax.clear()
    for i in range(signal.shape[2]):
        line, = ax.plot(signal[0, :, i], label=f'Channel {i+1}')
        lines.append(line)
    ax.set_title(f'Signal: {os.path.basename(file_path)} - {prediction}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()

    # Play audio only if the prediction has changed
    if prediction != previous_prediction:
        if prediction == "Seizure":
            play_audio('seizure.wav')
        else:
            play_audio('normal.wav')
        previous_prediction = prediction

# Create the animation
ani = FuncAnimation(fig, update, frames=len(csv_files), interval=200)

plt.show()
