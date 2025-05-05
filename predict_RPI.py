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
import time
from pydub.playback import play
from pydub import AudioSegment

# ========== CONFIGURATION ==========

# Check if running headless (no display)
HEADLESS = os.environ.get('DISPLAY', '') == ''
if HEADLESS:
    print('[INFO] Headless mode detected. Using Agg backend.')
    matplotlib.use('Agg')

# File paths
MODEL_PATH = "seizure_detection_model.h5"
DATASET_DIR = "dataset/"
NORMAL_AUDIO = "normal.WAV"
SEIZURE_AUDIO = "seizure.WAV"

# Blynk configuration
BLYNK_AUTH_TOKEN = "oiXcgQ7-lw9JBk_KqauFlEbW8O_BGIW7"
BLYNK_EVENT_NAME = "detect1"
BLYNK_EVENT_URL = f"https://blynk.cloud/external/api/logEvent?token={BLYNK_AUTH_TOKEN}&code={BLYNK_EVENT_NAME}"

# GPIO Configuration (Raspberry Pi)
GPIO_ENABLED = False
try:
    import RPi.GPIO as GPIO
    GPIO_ENABLED = True
    SEIZURE_ALERT_PIN = 17  # Change to your desired GPIO pin
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SEIZURE_ALERT_PIN, GPIO.OUT)
    print("[INFO] GPIO initialized successfully")
except ImportError:
    print("[WARNING] RPi.GPIO not available - running without GPIO functionality")

# ========== GPIO FUNCTIONS ==========

def glitter_gpio(duration=4, blink_speed=0.1):
    """Make GPIO pin glitter (rapid blink) for specified duration"""
    if not GPIO_ENABLED:
        return
        
    def _glitter():
        end_time = time.time() + duration
        while time.time() < end_time:
            GPIO.output(SEIZURE_ALERT_PIN, GPIO.HIGH)
            time.sleep(blink_speed)
            GPIO.output(SEIZURE_ALERT_PIN, GPIO.LOW)
            time.sleep(blink_speed)
    
    threading.Thread(target=_glitter, daemon=True).start()

# ========== BLYNK FUNCTION ==========

def trigger_blynk_event(description=None):
    """Trigger Blynk event using logEvent"""
    try:
        url = BLYNK_EVENT_URL
        if description:
            url += f"&description={requests.utils.quote(description)}"
        response = requests.get(url, timeout=5)
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
            if not os.path.exists(file):
                print(f"[ERROR] Audio file not found: {file}")
                return
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
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded.")
except Exception as e:
    raise RuntimeError(f"[ERROR] Failed to load model: {e}")

# ========== FEATURE EXTRACTION ==========

def extract_dwt_features(signals):
    """Extract Discrete Wavelet Transform features"""
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
    """Load EEG signal from CSV file"""
    try:
        signal_df = pd.read_csv(file_path)
        signal = signal_df.values
        return signal.reshape((1, signal.shape[0], signal.shape[1]))
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load CSV {file_path}: {e}")

def predict_seizure(file_path):
    """Make seizure prediction for a signal file"""
    try:
        signal = load_signal_from_csv(file_path)
        signal_features = extract_dwt_features(signal)
        prediction = model.predict(signal_features, verbose=0)
        label = "Seizure" if prediction[0][0] > 0.5 else "Normal"
        print(f"[INFO] {os.path.basename(file_path)} → Prediction: {label}")
        return label, signal
    except Exception as e:
        print(f"[ERROR] Prediction failed for {file_path}: {e}")
        return "Error", None

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    try:
        # Load dataset
        if not os.path.exists(DATASET_DIR):
            raise FileNotFoundError(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        
        csv_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) 
                    if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"[ERROR] No CSV files found in {DATASET_DIR}")

        # Check audio files
        for audio_file in [NORMAL_AUDIO, SEIZURE_AUDIO]:
            if not os.path.exists(audio_file):
                print(f"[WARNING] Audio file not found: {audio_file}")

        # Visualization setup
        fig, ax = plt.subplots(figsize=(10, 6))
        previous_prediction = None

        def update(frame):
            global previous_prediction
            file_path = random.choice(csv_files)
            prediction, signal = predict_seizure(file_path)
            
            if signal is None:
                return

            ax.clear()
            for i in range(signal.shape[2]):
                ax.plot(signal[0, :, i], label=f'Channel {i+1}')
            ax.set_title(f'{os.path.basename(file_path)} → {prediction}')
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right')

            # Trigger events only on prediction change
            if prediction != previous_prediction:
                if prediction == "Seizure":
                    play_audio(SEIZURE_AUDIO)
                    glitter_gpio(duration=4)  # Glitter for 4 seconds on seizure
                else:
                    play_audio(NORMAL_AUDIO)
                trigger_blynk_event(prediction)
                previous_prediction = prediction

        # Run animation
        ani = FuncAnimation(fig, update, frames=len(csv_files), interval=2000)
        if HEADLESS:
            if not shutil.which("ffmpeg"):
                raise RuntimeError("[ERROR] ffmpeg required. Install with: sudo apt install ffmpeg")
            print("[INFO] Saving animation...")
            ani.save("output.mp4", writer="ffmpeg", fps=2, dpi=100)
            print("[INFO] Animation saved as output.mp4")
        else:
            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down gracefully...")
    finally:
        if GPIO_ENABLED:
            GPIO.cleanup()
        print("[INFO] Cleanup complete")