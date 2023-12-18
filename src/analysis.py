import os
import copy
from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        lowcut: float = 1.0,
        highcut: float = 30.0,
        tmin: float = -0.2,
        tmax: float = 1.0,
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
        self.preprocess_eeg.filter(eeg, lowcut=lowcut, highcut=highcut)

        avg_evoked_list = []
        times_list = []
        for i in range(1, num_types + 1):
            avg_evoked, times = self.preprocess_eeg.epochs(
                eeg, events=events, event_id=i, tmin=tmin, tmax=tmax
            )
            avg_evoked_list.append(avg_evoked)
            times_list.append(times)
        return eeg, eeg_times, avg_evoked_list, times_list

    def analyze_erp_range(
        self,
        eeg_filename: str,
        event_filename: str,
        result_dir: str,
        num_types: int,
        cut: float = 13.0,
        freq_range: float = 2.0,
        tmin: float = 0.0,
        tmax: float = 1.0,
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

        self.preprocess_eeg.filter(eeg, lowcut=cut-freq_range, highcut=cut+freq_range)

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
        erd_lowcut: float = 8.0,
        erd_highcut: float = 11.0,
        ers_lowcut: float = 26.0,
        ers_highcut: float = 30.0,
        tmin: float = -4.0,
        tmax: float = 4.0,
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
        self.preprocess_eeg.filter(erd_eeg, lowcut=erd_lowcut, highcut=erd_highcut)  # ERD (Alpha)
        self.preprocess_eeg.filter(ers_eeg, lowcut=ers_lowcut, highcut=ers_highcut)  # ERS (Beta)

        # Squaring
        erd_eeg = self.preprocess_eeg.square(erd_eeg)
        ers_eeg = self.preprocess_eeg.square(ers_eeg)

        # Smoothing
        erd_eeg = self.preprocess_eeg.moving_average(erd_eeg)
        ers_eeg = self.preprocess_eeg.moving_average(ers_eeg)

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
        erds_lowcut: float = 8.0,
        erds_highcut: float = 12.0,
        erds_whole_lowcut: float = 8.0,
        erds_whole_highcut: float = 30.0,
        tmin: float = -4.0,
        tmax: float = 4.0,
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
        self.preprocess_eeg.filter(erds_eeg, lowcut=erds_lowcut, highcut=erds_highcut)  # ERDS (Mu)
        self.preprocess_eeg.filter(erds_whole_eeg, lowcut=erds_whole_lowcut, highcut=erds_whole_highcut)  # ERDS_whole

        # Squaring
        erds_eeg = self.preprocess_eeg.square(erds_eeg)
        erds_whole_eeg = self.preprocess_eeg.square(erds_whole_eeg)

        # Smoothing
        erds_eeg = self.preprocess_eeg.moving_average(erds_eeg)
        erds_whole_eeg = self.preprocess_eeg.moving_average(erds_whole_eeg)

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
        early_cut: int = 22,
        scale: str = "min_max"
    ):
        # Check result directory
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # Read eeg and events
        fft = pd.read_csv(fft_filename)
        fft = fft.iloc[early_cut:, :]
        
        # avg_values = {f"{float(frequency):.2f}Hz": fft.loc[:, f"{float(frequency-freq_range):.2f}Hz":f"{float(frequency+freq_range):.2f}Hz"].mean(axis=1) for frequency in frequencies
        # }
        # avg_df = pd.DataFrame(avg_values)

        # if scale == "min_max":
        #     scaler = MinMaxScaler(feature_range=(0,1))
        #     scaled_avg_df = pd.DataFrame(scaler.fit_transform(avg_df), columns=avg_df.columns)
        #     scaled_avg_df["Time"] = pd.to_datetime(fft["Time"])
        #     scaled_avg_df.set_index("Time", inplace=True)
        #     return scaled_avg_df
        # elif scale == "standard":
        #     scaler = StandardScaler()
        #     scaled_avg_df = pd.DataFrame(scaler.fit_transform(avg_df), columns=avg_df.columns)
        #     scaled_avg_df["Time"] = pd.to_datetime(fft["Time"])
        #     scaled_avg_df.set_index("Time", inplace=True)
        #     return scaled_avg_df
        # elif scale == "original":
        #     avg_df["Time"] = pd.to_datetime(fft["Time"])
        #     avg_df.set_index("Time", inplace=True)
        #     return avg_df
        # else:
        #     raise ValueError("Invalid scale type")

        # harmonic_frequencies = []
        # for i in range(1, 4):
        #     harmonic_frequencies += [frequency * i for frequency in frequencies]
        
        # avg_values = {f"{float(frequency):.2f}Hz": fft.loc[:, f"{float(frequency-freq_range):.2f}Hz":f"{float(frequency+freq_range):.2f}Hz"].mean(axis=1) for frequency in harmonic_frequencies
        # }
        # avg_df = pd.DataFrame(avg_values)
        
        # avg_values = {f"{float(frequency):.2f}Hz": fft.loc[:, f"{float(frequency-freq_range):.2f}Hz":f"{float(frequency+freq_range):.2f}Hz"].mean(axis=1) for frequency in frequencies
        # }
        # avg_df = pd.DataFrame(avg_values)

        avg_dfs = []
        for i in range(1, 4):
            avg_values = {f"{float(frequency):.2f}Hz": fft.loc[:, f"{float(frequency*i-freq_range):.2f}Hz":f"{float(frequency*i+freq_range):.2f}Hz"].mean(axis=1) for frequency in frequencies
            }
            # avg_values = {}
            # avg_values_0 = {f"{float(frequencies[0]):.2f}Hz": fft.loc[:, f"{float(frequencies[0]*i-freq_range):.2f}Hz":f"{float(frequencies[0]*i+freq_range):.2f}Hz"].mean(axis=1)}
            # avg_values_1 = {f"{float(frequencies[1]):.2f}Hz": fft.loc[:, f"{float(frequencies[1]*i+1-freq_range):.2f}Hz":f"{float(frequencies[1]*i+1+freq_range):.2f}Hz"].mean(axis=1)}
            # avg_values_2 = {f"{float(frequencies[2]):.2f}Hz": fft.loc[:, f"{float(frequencies[2]*i+1-freq_range):.2f}Hz":f"{float(frequencies[2]*i+1+freq_range):.2f}Hz"].mean(axis=1)}
            # avg_values_3 = {f"{float(frequencies[3]):.2f}Hz": fft.loc[:, f"{float(frequencies[3]*i-freq_range):.2f}Hz":f"{float(frequencies[3]*i+freq_range):.2f}Hz"].mean(axis=1)}
            # avg_values.update(avg_values_0)
            # avg_values.update(avg_values_1)
            # avg_values.update(avg_values_2)
            # avg_values.update(avg_values_3)
            avg_df = pd.DataFrame(avg_values)
            avg_dfs.append(avg_df)
            
        if scale == "min_max":
            scaled_avg_values = 0
            for avg_df in avg_dfs:
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_avg_values += scaler.fit_transform(avg_df)
            scaled_avg_df = pd.DataFrame(scaled_avg_values, columns=avg_dfs[0].columns)
            scaled_avg_df["Time"] = pd.to_datetime(fft["Time"])
            scaled_avg_df.set_index("Time", inplace=True)
            return scaled_avg_df
        elif scale == "standard":
            scaled_avg_values = 0
            for avg_df in avg_dfs:
                scaler = StandardScaler()
                scaled_avg_values += scaler.fit_transform(avg_df)
            scaled_avg_df = pd.DataFrame(scaled_avg_values, columns=avg_dfs[0].columns)
            scaled_avg_df["Time"] = pd.to_datetime(fft["Time"])
            scaled_avg_df.set_index("Time", inplace=True)
        elif scale == "original":
            avg_df["Time"] = pd.to_datetime(fft["Time"])
            avg_df.set_index("Time", inplace=True)
            return avg_df
        else:
            raise ValueError("Invalid scale type")