import os
import copy
from typing import List

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
        if not os.path.isdir(os.path.join(os.getcwd(), result_dir)):
            os.mkdir(os.path.join(os.getcwd(), result_dir))

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
        if not os.path.isdir(os.path.join(os.getcwd(), result_dir)):
            os.mkdir(os.path.join(os.getcwd(), result_dir))

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
