from typing import List
import copy

import numpy as np
import pandas as pd

from .iir import FilterSignal
from sklearn.decomposition import FastICA


class PreprocessEEG:
    def __init__(self, channels: List, fs: int):
        self.filter_signal = FilterSignal(fs=fs, order=2)
        self.channels = channels
        self.fs = fs

    def read_eeg(self, filename: str):
        eeg = np.loadtxt(filename, skiprows=1, delimiter=",")
        eeg = np.transpose(eeg)
        times = [i / float(self.fs) for i in range(len(eeg[0]))]
        return eeg, times

    def read_events(self, filename: str):
        data = pd.read_csv(filename)
        stimuli = data["Stimulus"]
        indices = data["ISI"].cumsum() + data["RT"].cumsum() - data["RT"][0]
        events = []
        for i in range(len(indices)):
            events.append([int(indices[i] * self.fs / 1000.0), 0, stimuli[i]])
        return events

    def extract_eeg_each_channel(self, eeg):
        eeg_each_channel = []
        for i in range(len(self.channels)):
            remove_channels = []
            for j in range(len(self.channels)):
                if i != j:
                    remove_channels.append(self.channels[j])
            eeg_tmp = copy.deepcopy(eeg)
            eeg_tmp.drop_channels(remove_channels)
            eeg_each_channel.append(eeg_tmp)
        return eeg_each_channel

    def synchronize_time_interval(self, events, eeg_start_tm, event_start_tm):
        # Parse eeg timestamp
        eeg_t_tokens = np.array(eeg_start_tm.split(".")).astype("float")
        eeg_start_tm = (
            eeg_t_tokens[0] * 60 * 60 * 1000
            + eeg_t_tokens[1] * 60 * 1000
            + eeg_t_tokens[2] * 1000
        )
        # Parse event timestmap
        event_t_tokens = np.array(event_start_tm.split(".")).astype("float")
        event_start_tm = (
            event_t_tokens[0] * 60 * 60 * 1000
            + event_t_tokens[1] * 60 * 1000
            + event_t_tokens[2] * 1000
            + int(str(event_t_tokens[3])[:3])
        )
        # Calculate time interval between eeg and events
        time_interval = event_start_tm - eeg_start_tm

        # Synchronize
        for i in range(len(events)):
            events[i][0] += int(time_interval)

        return events

    def epochs(self, eeg, events, event_id, tmin: float, tmax: float):
        evoked = []
        times = []
        tmin_idx = tmin * self.fs
        tmax_idx = tmax * self.fs

        # Create time indices
        for i in range(int((tmax - tmin) * self.fs) + 1):
            time = np.round(tmin + (i * 1.0 / self.fs), 2)
            times.append(time)

        # Loop for events
        for event in events:
            # Check event id
            id = event[2]
            if id != event_id:
                continue

            # Calculate index
            idx = int(event[0] / 1000.0 * self.fs)
            start_idx = int(idx + tmin_idx)
            end_idx = int(idx + tmax_idx)
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(eeg[0]):
                break

            # Crop evoked based on event
            erp = eeg[:, start_idx:end_idx]

            # Add to evoked
            if len(erp[0]) == end_idx - start_idx:
                evoked.append(erp)
        # Average
        avg_evoked = np.average(evoked, axis=0)

        return avg_evoked, times

    def filter(self, eeg, lowcut: int, highcut: int):
        for i in range(len(eeg)):
            eeg[i] = self.filter_signal.butter_bandpass_filter(eeg[i], lowcut, highcut)

    def normalize(self, eeg):
        eeg *= 10000
        return eeg

    def ica(self, evoked, n_components=3):
        fast_ica = FastICA(n_components=n_components)
        S_ = fast_ica.fit_transform(np.transpose(evoked))  # Reconstruct signals
        A_ = fast_ica.mixing_  # Get estimated mixing matrix
        return np.transpose(S_), np.transpose(A_)

    def square(self, eeg):
        return eeg**2

    def moving_average(self, signal):
        window_size = self.fs
        ma_signal = []
        for s in signal:
            ma = []
            half_window_size = int(window_size / 2)
            # First index
            ma.append(s[0])
            # First half window size
            for i in range(1, half_window_size):
                ma.append(np.average(s[0:i]))
            # Window size
            for i in range(half_window_size, len(s) - half_window_size):
                ma.append(np.average(s[i - half_window_size : i + half_window_size]))
            # Last half window size
            for i in range(len(s) - half_window_size, len(s) - 1):
                ma.append(np.average(s[i:]))
            # Last index
            ma.append(s[-1])
            ma_signal.append(ma)
        ma_signal = np.array(ma_signal)
        return ma_signal
