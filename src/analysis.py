import os
import copy
from typing import List

import pandas as pd

from .preprocess import PreprocessEEG


class AnalyzeEEG:
    def __init__(self, channels: List, fs: int):
        self.preprocess_eeg = PreprocessEEG(channels, fs)

    def analyze_erp(
        self,
        eeg_filename: str,
        event_filename: str,
        result_dir: str,
        num_types: int,
    ):
        # Check result directory
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # Read eeg and events
        eeg, eeg_times = self.preprocess_eeg.read_eeg(eeg_filename)
        eeg = self.preprocess_eeg.normalize(eeg)  # Normalize
        events = self.preprocess_eeg.read_events(event_filename)

        # Synchronize time interval
        eeg_start_tm = eeg_filename.split("_")[-1].replace(".csv", "")
        event_start_tm = event_filename.split("_")[-1].replace(".csv", "")
        events = self.preprocess_eeg.synchronize_time_interval(
            events, eeg_start_tm, event_start_tm
        )

        # Apply filter (1-30 Hz)
        self.preprocess_eeg.filter(eeg, lowcut=1, highcut=30)

        # Analysis ERP
        tmin, tmax = -0.2, 1.0

        avg_evoked_list = []
        times_list = []
        for i in range(1, num_types + 1):
            avg_evoked, times = self.preprocess_eeg.epochs(
                eeg, events=events, event_id=i, tmin=tmin, tmax=tmax
            )
            avg_evoked_list.append(avg_evoked)
            times_list.append(times)
        return eeg, eeg_times, avg_evoked_list, times_list

    def analyze_erds(
        self,
        eeg_filename: str,
        event_filename: str,
        result_dir: str,
        num_types: int,
    ):
        # Check result directory
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # Read eeg and events
        eeg, eeg_times = self.preprocess_eeg.read_eeg(eeg_filename)
        events = self.preprocess_eeg.read_events(event_filename)

        # Synchronize time interval
        eeg_start_tm = eeg_filename.split("_")[-1].replace(".csv", "")
        event_start_tm = event_filename.split("_")[-1].replace(".csv", "")
        events = self.preprocess_eeg.synchronize_time_interval(
            events, eeg_start_tm, event_start_tm
        )

        # Apply filter
        erd_eeg = copy.deepcopy(eeg)
        ers_eeg = copy.deepcopy(eeg)
        self.preprocess_eeg.filter(erd_eeg, lowcut=8, highcut=11)  # ERD (Alpha)
        self.preprocess_eeg.filter(ers_eeg, lowcut=26, highcut=30)  # ERS (Beta)

        # Squaring
        erd_eeg = self.preprocess_eeg.square(erd_eeg)
        ers_eeg = self.preprocess_eeg.square(ers_eeg)

        # Smoothing
        erd_eeg = self.preprocess_eeg.moving_average(erd_eeg)
        ers_eeg = self.preprocess_eeg.moving_average(ers_eeg)

        # Analysis evoked potential
        tmin, tmax = -4.0, 4.0

        erd_avg_evoked_list = []
        erd_times_list = []
        for i in range(1, num_types + 1):
            erd_avg_evoked, erd_times = self.preprocess_eeg.epochs(
                erd_eeg, events=events, event_id=i, tmin=tmin, tmax=tmax
            )
            erd_avg_evoked_list.append(erd_avg_evoked)
            erd_times_list.append(erd_times[:-1])

        ers_avg_evoked_list = []
        ers_times_list = []
        for i in range(1, num_types + 1):
            ers_avg_evoked, ers_times = self.preprocess_eeg.epochs(
                ers_eeg, events=events, event_id=i, tmin=tmin, tmax=tmax
            )
            ers_avg_evoked_list.append(ers_avg_evoked)
            ers_times_list.append(ers_times[:-1])
        return (
            eeg,
            eeg_times,
            erd_avg_evoked_list,
            erd_times_list,
            ers_avg_evoked_list,
            ers_times_list,
        )

    def analyze_whole_erds(
        self,
        eeg_filename: str,
        event_filename: str,
        result_dir: str,
    ):
        # Check result directory
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # Read eeg and events
        eeg, eeg_times = self.preprocess_eeg.read_eeg(eeg_filename)
        events = self.preprocess_eeg.read_events(event_filename)

        # Synchronize time interval
        eeg_start_tm = eeg_filename.split("_")[-1].replace(".csv", "")
        event_start_tm = event_filename.split("_")[-1].replace(".csv", "")
        events = self.preprocess_eeg.synchronize_time_interval(
            events, eeg_start_tm, event_start_tm
        )

        # Apply filter
        erds_eeg = copy.deepcopy(eeg)
        erds_whole_eeg = copy.deepcopy(eeg)
        self.preprocess_eeg.filter(erds_eeg, lowcut=8, highcut=12)  # ERD (Alpha)
        self.preprocess_eeg.filter(erds_whole_eeg, lowcut=8, highcut=30)  # ERS (Beta)

        # Squaring
        erds_eeg = self.preprocess_eeg.square(erds_eeg)
        erds_whole_eeg = self.preprocess_eeg.square(erds_whole_eeg)

        # Smoothing
        erds_eeg = self.preprocess_eeg.moving_average(erds_eeg)
        erds_whole_eeg = self.preprocess_eeg.moving_average(erds_whole_eeg)

        # Analysis evoked potential
        tmin, tmax = -4.0, 4.0

        erds_avg_evoked_list = []
        erds_times_list = []
        erds_avg_evoked, erds_times = self.preprocess_eeg.epochs(
            erds_eeg, events=events, event_id=0, tmin=tmin, tmax=tmax
        )
        erds_avg_evoked_list.append(erds_avg_evoked)
        erds_times_list.append(erds_times[:-1])

        erds_whole_avg_evoked_list = []
        erds_whole_times_list = []
        erds_whole_avg_evoked, erds_whole_times = self.preprocess_eeg.epochs(
            erds_whole_eeg, events=events, event_id=0, tmin=tmin, tmax=tmax
        )
        erds_whole_avg_evoked_list.append(erds_whole_avg_evoked)
        erds_whole_times_list.append(erds_whole_times[:-1])
        return (
            eeg,
            eeg_times,
            erds_avg_evoked_list,
            erds_times_list,
            erds_whole_avg_evoked_list,
            erds_whole_times_list,
        )

    def analyze_ssvep(
        self,
        fft_filename: str,
        frequencies: List,
        freq_range: float,
        result_dir: str,
    ):
        # Check result directory
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # Read eeg and events
        fft = pd.read_csv(fft_filename)
        fft = fft.iloc[22:, :]
        avg_values = {f"{float(frequency):.2f}Hz": fft.loc[:, f"{float(frequency-freq_range):.2f}Hz":f"{float(frequency+freq_range):.2f}Hz"].mean(axis=1) for frequency in frequencies
        }
        avg_df = pd.DataFrame(avg_values)
        avg_df["Time"] = pd.to_datetime(fft["Time"])
        avg_df.set_index("Time", inplace=True)
        return avg_df