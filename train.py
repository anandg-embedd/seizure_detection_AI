##import kagglehub
##
### Download latest version
##path = kagglehub.dataset_download("adibadea/chbmitseizuredataset")
##
##print("Path to dataset files:", path)

##import numpy as np
##
### Load the dataset
##file_path = "dataset/eeg-seizure_train.npz"
##data = np.load(file_path)
##
### Inspect the contents
##print(data.files)  # List of arrays in the .npz file
##
### Access and print the arrays
##train_signals = data['train_signals']
##train_labels = data['train_labels']
##
##print("Train Signals:", train_signals)
##print("Train Labels:", train_labels)

# import numpy as np
# import os

# # Load the dataset
# file_path = "dataset/eeg-seizure_train.npz"
# data = np.load(file_path)
# train_signals = data['train_signals']
# train_labels = data['train_labels']

# # Print the shape of train_signals and train_labels
# print("Shape of train_signals:", train_signals.shape)
# print("Shape of train_labels:", train_labels.shape)


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Input, Reshape
from sklearn.model_selection import train_test_split
import pywt

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

# Load the dataset
file_path = "dataset/eeg-seizure_train.npz"
data = np.load(file_path)
train_signals = data['train_signals']
train_labels = data['train_labels']

# Log the shape of the loaded data
print("Shape of train_signals:", train_signals.shape)
print("Shape of train_labels:", train_labels.shape)

# Reduce the dataset size by taking a subset (e.g., 10%)
subset_size = int(0.1 * train_signals.shape[0])
train_signals = train_signals[:subset_size]
train_labels = train_labels[:subset_size]

# Log the shape after taking a subset
print("Shape of train_signals after subset:", train_signals.shape)
print("Shape of train_labels after subset:", train_labels.shape)

# Extract DWT features from the signals
train_features = extract_dwt_features(train_signals)

# Log the shape of the extracted features
print("Shape of extracted features:", train_features.shape)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(train_features, train_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Log the shapes after splitting the data
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)

# Reshape the data for the RNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Log the shapes after reshaping
print("Shape of X_train after reshaping:", X_train.shape)
print("Shape of X_val after reshaping:", X_val.shape)
print("Shape of X_test after reshaping:", X_test.shape)

# Build the improved RNN model with Conv1D layers and additional LSTM layers
# Define the model
model = Sequential()

# Add the first convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(23, 282)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

# Add the second convolutional layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

# Add an LSTM layer
model.add(LSTM(100))

# Add a dense output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save("seizure_detection_model.h5")

print("Model training complete and saved as seizure_detection_model.h5")
