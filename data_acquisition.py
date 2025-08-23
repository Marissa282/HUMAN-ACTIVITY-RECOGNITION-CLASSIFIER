#------------------------------------------------------------------------------------------------------------------
#   Mobile sensor data acquisition example
#------------------------------------------------------------------------------------------------------------------
import time
import requests
import numpy as np
import threading

import random
import pickle
from datetime import datetime

from scipy.interpolate import interp1d

# Experiment configuration
conditions = [
    ('Jumping', 1),
    ('Squats', 2),
    ('Lateral Lunges', 3),
    ('Walking straight', 4),
    ('Nothing', 5),
    ('Hip Circles', 6)
]

n_trials = 10                # Number of trials per condition
n_windows = 10              # Number of windows for each trial

fixation_cross_time = 2     # Time in seconds for attention fixation
preparation_time = 1        # Time in seconds for preparation before each trial
window_time = 0.5           # Time in seconds for each trial window
rest_time = 5               # Time in seconds for rest between trials

sampling_rate = 30          # Sampling rate in Hz of the output data
max_samp_rate = 5000        # Maximum possible sampling rate
max_window_samples = int(window_time*max_samp_rate)     # Maximum number of samples in each window

trials = n_trials * conditions
random.shuffle(trials)

trial_time = fixation_cross_time + preparation_time + n_windows * window_time + rest_time

# Communication parameters
IP_ADDRESS = '192.168.1.69'
COMMAND = 'accX&accY&accZ&acc_time'
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data buffer
n_signals = 3   # Number of signals (accX, accY, accZ)
buffer_size = int(2 * len(trials) * trial_time * max_samp_rate)

buffer = np.zeros((buffer_size, n_signals + 1), dtype='float64')
buffer_index = 0

# Flag for stopping the data acquisition
stop_recording_flag = threading.Event()

# Mutex for thread-safe access to the buffer
buffer_lock = threading.Lock()

# Function for continuously fetching data from the mobile device
def fetch_data():
    sleep_time = 1. / max_samp_rate
    while not stop_recording_flag.is_set():
        try:
            response = requests.get(BASE_URL, timeout=0.5)
            response.raise_for_status()
            data = response.json()

            global buffer, buffer_index

            with buffer_lock:
                buffer[buffer_index:, 0] = data["buffer"]["acc_time"]["buffer"][0]
                buffer[buffer_index:, 1] = data["buffer"]["accX"]["buffer"][0]
                buffer[buffer_index:, 2] = data["buffer"]["accY"]["buffer"][0]
                buffer[buffer_index:, 3] = data["buffer"]["accZ"]["buffer"][0]

                buffer_index += 1

        except Exception as e:
            print(f"Error fetching data: {e}")

        time.sleep(sleep_time)

def stop_recording():
    stop_recording_flag.set()
    recording_thread.join()

# Start data acquisition
recording_thread = threading.Thread(target=fetch_data, daemon=True)
recording_thread.start()

# Run experiment
print("********* Experiment in progress *********")
time.sleep(fixation_cross_time)

window_info = []
count = 0
for t in trials:
    count += 1
    print("\n********* Trial {}/{} *********".format(count, len(trials)))
    time.sleep(fixation_cross_time)

    print(t[0])
    time.sleep(preparation_time)

    for window in range(n_windows):
        time.sleep(window_time)
        window_info.append((t[0], t[1], buffer_index))

    print("----Rest----")
    time.sleep(rest_time)

# Stop data acquisition
stop_recording()

# Calculate average sampling rate
t = buffer[:buffer_index, 0]
diff_t = np.diff(t)

print("Min sampling rate: {:.2f} Hz".format(1. / np.max(diff_t)))
print("Max sampling rate: {:.2f} Hz".format(1. / np.min(diff_t)))
print("Average sampling rate: {:.2f} Hz".format(1. / np.mean(diff_t)))

# Interpolation functions for uniform sampling
interp_x1 = interp1d(t, buffer[:buffer_index, 1], kind='linear', fill_value="extrapolate")
interp_x2 = interp1d(t, buffer[:buffer_index, 2], kind='linear', fill_value="extrapolate")
interp_x3 = interp1d(t, buffer[:buffer_index, 3], kind='linear', fill_value="extrapolate")

# Separate the data for each trial
window_samples = int(sampling_rate * window_time)
data = []
for w in window_info:
    condition = w[0]
    condition_id = w[1]
    start_index = w[2]

    t_start = buffer[start_index, 0]
    t_uniform = np.linspace(t_start, t_start + window_time, int(window_time * sampling_rate))

    signal_data = np.column_stack((interp_x1(t_uniform), interp_x2(t_uniform), interp_x3(t_uniform)))
    data.append((condition, condition_id, signal_data))

# Save data
now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
outputFile = open(now + '.obj', 'wb')
pickle.dump(data, outputFile)
outputFile.close()