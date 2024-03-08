from typing import Tuple

import numpy as np

from scipy.signal import butter, lfilter


class FilterSignal:
    def __init__(
        self, 
        fs: int, 
        order: int,
    ) -> None:
        self.fs = fs
        self.order = order

    def filtered_signal(
        self, 
        filter: np.ndarray,
    ) -> np.ndarray:
        filtered_signal = []
        for i in range(0, len(filter)):
            filtered_signal.append(float(filter[i]))
        return np.array(filtered_signal)

    def butter_bandpass(
        self, 
        lowcut: float, 
        highcut: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(self.order, [low, high], btype="band")
        return b, a

    def butter_bandpass_filter(
        self, 
        data: np.ndarray, 
        lowcut: float, 
        highcut: float,
    ) -> np.ndarray:
        b, a = self.butter_bandpass(lowcut, highcut)
        y = lfilter(b, a, data)
        return self.filtered_signal(y)

    def butter_lowpass(
        self, 
        cut: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * self.fs
        ncut = cut / nyq
        b, a = butter(self.order, ncut, btype="low", analog=True)
        return b, a

    def butter_lowpass_filter(
        self, 
        data: np.ndarray, 
        cut: float,
    ) -> np.ndarray:
        b, a = self.butter_lowpass(cut)
        y = lfilter(b, a, data)
        return self.filtered_signal(y)

    def butter_highpass(
        self, 
        cut: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        nyq = 0.5 * self.fs
        ncut = cut / nyq
        b, a = butter(self.order, ncut, btype="high", analog=True)
        return b, a

    def butter_highpass_filter(
        self, 
        data: np.ndarray, 
        cut: float,
    ) -> np.ndarray:
        b, a = self.butter_highpass(cut)
        y = lfilter(b, a, data)
        return self.filtered_signal(y)
