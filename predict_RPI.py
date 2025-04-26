import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pywt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import threading
import requests
import shutil
from pydub.playback import play
from pydub import AudioSegment

# ========== CONFIGURATION ==========

# Check if running headless (no display)
HEADLESS = os.environ.get('DISPLAY', '') == ''
if HEADLESS:
    print('[INFO] Headless mode detected. Using Agg backend.')
    matplotlib.use('Agg')

# File paths
MODEL_PATH = "seizure detection_model.h5"
DATASET_DIR = "dataset/"
NORMAL_AUDIO = "normal.WAV"
SEIZURE_AUDIO = "seizure.WAV"

# Blynk configuration
BLYNK_AUTH_TOKEN = "3hwBdWBZOYwjkxG2Plu0McIZnDBZHT4f"
BLYNK_EVENT_NAME = "info1"  # Event you created in the Blynk dashboard
BLYNK_EVENT_URL = f"https://blynk.cloud/external/api/logEvent?token={BLYNK_AUTH_TOKEN}&code={BLYNK_EVENT_NAME}&description="

# ========== BLYNK FUNCTION ==========

def trigger_blynk_event(description=None):
    """Trigger Blynk event using logEvent"""
    try:
        url = f"https://blynk.cloud/external/api/logEvent?token={BLYNK_AUTH_TOKEN}&code={BLYNK_EVENT_NAME}"
        if description:
            url += f"&description={requests.utils.quote(description)}"
        response = requests.get(url)
        if response.status_code == 200:
            print(f"[INFO] Blynk event triggered: {BLYNK_EVENT_NAME}")
        else:
            print(f"[ERROR] Failed to trigger Blynk event: {response.status_code} — {response.text}")
    except Exception as e:
        print(f"[ERROR] Exception triggering Blynk event: {e}")


# ========== AUDIO FUNCTION ==========

audio_lock = threading.Lock()

def play_audio(file):
    """Play an audio file asynchronously"""
    def _play():
        try:
            audio = AudioSegment.from_file(file)
            with audio_lock:
                play(audio)
        except Exception as e:
            print(f"[ERROR] Audio playback failed: {e}")
    threading.Thread(target=_play, daemon=True).start()

# ========== LOAD MODEL ==========

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model not found at {MODEL_PATH}")
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded.")

# ========== FEATURE EXTRACTION ==========

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

# ========== SIGNAL PROCESSING ==========

def load_signal_from_csv(file_path):
    signal_df = pd.read_csv(file_path)
    signal = signal_df.values
    return signal.reshape((1, signal.shape[0], signal.shape[1]))

def predict_seizure(file_path):
    signal = load_signal_from_csv(file_path)
    signal_features = extract_dwt_features(signal)
    prediction = model.predict(signal_features, verbose=0)
    label = "Seizure" if prediction[0][0] > 0.5 else "Normal"
    print(f"[INFO] {os.path.basename(file_path)} → Prediction: {label}")
    return label, signal

# ========== LOAD DATASET ==========

csv_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"[ERROR] No CSV files found in {DATASET_DIR}")

# ========== VISUALIZATION ==========

fig, ax = plt.subplots()
previous_prediction = None

def update(frame):
    global previous_prediction
    file_path = random.choice(csv_files)
    prediction, signal = predict_seizure(file_path)

    ax.clear()
    for i in range(signal.shape[2]):
        ax.plot(signal[0, :, i], label=f'Channel {i+1}')
    ax.set_title(f'{os.path.basename(file_path)} → {prediction}')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right')

    # Trigger events only on prediction change
    if prediction != previous_prediction:
        audio_file = SEIZURE_AUDIO if prediction == "Seizure" else NORMAL_AUDIO
        play_audio(audio_file)
		trigger_blynk_event(prediction)
        previous_prediction = prediction

# ========== RUN ANIMATION ==========

ani = FuncAnimation(fig, update, frames=len(csv_files), interval=200)
plt.show()

if HEADLESS:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("[ERROR] ffmpeg required. Install it using: sudo apt install ffmpeg")
    ani.save("output.mp4", writer="ffmpeg", fps=5)
    print("[INFO] Headless mode: Animation saved as output.mp4")
else:
    plt.show()