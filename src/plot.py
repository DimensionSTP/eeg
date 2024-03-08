import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class PlotEEG:
    def __init__(
        self,
        channels: List[str],
        result_dir: str,
        is_show: bool,
        is_save: bool,
        eeg: np.ndarray,
        eeg_times: np.ndarray,
        eeg_filename: str,
    ) -> None:
        self.channels = channels
        self.result_dir = result_dir
        self.is_show = is_show
        self.is_save = is_save
        self.eeg = eeg
        self.eeg_times = eeg_times
        self.eeg_filename = eeg_filename

    def init_plots(
        self, 
        n_plots: int, 
        width: int=6, 
        height: float=2.5, 
        is_line: bool=False,
    ) -> Tuple[Figure, Axes]:
        fig, axarr = plt.subplots(
            n_plots, sharex=True, figsize=(width, height * n_plots)
        )
        plt.tight_layout()
        if is_line:
            plt.axhline(y=0, color="black", linewidth=1)
            plt.axvline(x=0, color="r", linestyle="--", linewidth=1)
        return fig, axarr

    def plot_data(
        self,
        axis: Axes,
        x: np.ndarray,
        y: np.ndarray,
        title: str="",
        fontsize: int=10,
        labelsize: int=8,
        color: str="salmon",
        xlabel: str="",
        ylabel: str="",
        linewidth: float=0.5,
    ) -> None:
        axis.set_title(title, fontsize=fontsize)
        axis.grid(which="both", axis="both", linestyle="--")
        axis.tick_params(labelsize=labelsize)
        axis.set_xlabel(xlabel, fontsize=labelsize + 1)
        axis.set_ylabel(ylabel, fontsize=labelsize + 1)
        axis.autoscale(tight=True)
        axis.plot(x, y, color=color, zorder=1, linewidth=linewidth)

    def plot_points(
        self, 
        axis: Axes, 
        values: np.ndarray, 
        indices: List[int],
    ) -> None:
        axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)

    def show_plot(self) -> None:
        plt.show()
        plt.clf()
        plt.close()

    def save_plot(
        self, 
        fig: Figure, 
        filename: str,
    ) -> None:
        fig.savefig(filename)

    def clean_plot(self) -> None:
        plt.clf()
        plt.close()

    def plot_eeg(self) -> None:
        fig, axarr = self.init_plots(1)
        for i in range(len(self.eeg)):  # For electrode
            self.plot_data(
                axis=axarr,
                x=self.eeg_times,
                y=self.eeg[i],
                color="black",
                xlabel="Time (s)",
                ylabel="uV",
                title="EEG (" + str(len(self.eeg)) + " channels)",
            )
        if self.is_save:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.save_plot(fig, filename=f"{self.result_dir}/{self.eeg_filename}.png")
        if self.is_show:
            self.show_plot()
        self.clean_plot()

    def plot_electrode(
        self,
        avg_evoked: List[np.ndarray],
        times: List[np.ndarray],
        filename: str,
    ) -> None:
        for i in range(len(avg_evoked)):  # For electrode
            fig, axarr = self.init_plots(1, is_line=True)
            self.plot_data(
                axis=axarr,
                x=times,
                y=avg_evoked[i],
                color="black",
                xlabel="Time (s)",
                ylabel="uV",
                title="EEG ("
                + str(len(avg_evoked))
                + " channels) - "
                + self.channels[i]
                + " Electrode",
            )
            if self.is_save:
                if not os.path.exists(self.result_dir):
                    os.makedirs(self.result_dir)
                self.save_plot(
                    fig,
                    filename=f"{self.result_dir}/{filename}_average_{self.channels[i]}.png",
                )
            if self.is_show:
                self.show_plot()
            self.clean_plot()


def plot_ssvep(
    df: pd.DataFrame,
    save_path: str,
    figsize: tuple = (15,8),
) -> None:
    plt.figure(figsize=figsize)
    for freq, values in df.items():
        plt.plot(df.index, values, label=f"Average around {freq}")
    
    plt.xlabel("Time")
    plt.ylabel("Average FFT Values")
    plt.title("Average FFT Values over Time for Selected Frequencies")
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(save_path)