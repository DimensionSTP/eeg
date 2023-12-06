import os
from typing import List
from datetime import datetime

import pandas as pd

from src.task import selection_task
from src.analysis import AnalyzeEEG
from src.plot import PlotEEG, plot_ssvep
from src.recommendation import recommend_selection


def erp_ssvep_select(
    screen_width: int,
    screen_height: int,
    fs: int,
    channels: List,
    isi: int,
    image_folder: str,
    frequencies: List,
    experiment_duration: int,
    event_save_path: str,
    result_dir: str,
    freq_range: float = 5.0,
    tmin: float = 0.0,
    tmax: float = 1.0,
    early_cut: int = 0,
    scaled: bool = True,
):
    today = str(datetime.now().date())
    if not os.path.exists(f"./data/{today}"):
        os.makedirs(f"./data/{today}")
    if not os.path.exists(f"./event/{today}"):
        os.makedirs(f"./event/{today}")

    selection_task(
        screen_width=screen_width,
        screen_height=screen_height,
        isi=isi,
        image_folder=image_folder,
        frequencies=frequencies,
        experiment_duration=experiment_duration,
        event_save_path=f"{event_save_path}/{today}",
    )

    rawdata_folders = os.listdir("C:/MAVE_RawData")

    text_file_name = f"C:/MAVE_RawData/{rawdata_folders[-1]}/Rawdata.txt"
    data_df = pd.read_csv(text_file_name, delimiter="\t")

    record_start_time = data_df.iloc[0, 0]
    hour = str(record_start_time).split(":")[0]
    min = str(record_start_time).split(":")[1]
    sec = str(record_start_time).split(":")[2].split(".")[0]

    data_df = data_df[channels]
    data_file_path = f"./data/{today}/Rawdata_{hour}.{min}.{sec}.csv"
    data_df.to_csv(data_file_path, index=False)

    event_paths = os.listdir(f"./event/{today}")
    event_file = f"./event/{today}/{event_paths[-1]}"

    fp1_file_name = f"C:/MAVE_RawData/{rawdata_folders[-1]}/Fp1_FFT.txt"
    fp1_df = pd.read_csv(fp1_file_name, delimiter="\t")
    fp2_file_name = f"C:/MAVE_RawData/{rawdata_folders[-1]}/Fp2_FFT.txt"
    fp2_df = pd.read_csv(fp2_file_name, delimiter="\t")

    record_start_time = fp1_df.iloc[0, 0]
    hour = str(record_start_time).split(":")[0]
    min = str(record_start_time).split(":")[1]
    sec = str(record_start_time).split(":")[2].split(".")[0]

    fp1_file_path = f"./data/{today}/fp1_{hour}.{min}.{sec}.csv"
    fp1_df.to_csv(fp1_file_path, index=False)
    fp2_file_path = f"./data/{today}/fp2_{hour}.{min}.{sec}.csv"
    fp2_df.to_csv(fp2_file_path, index=False)

    analyze_eeg = AnalyzeEEG(channels=channels, fs=fs)
    
    avg_evoked_list_frequencies = []
    for num in range(len(frequencies)):
        eeg, eeg_times, avg_evoked_list, times_list = analyze_eeg.analyze_erp_range(
            eeg_filename=data_file_path,
            event_filename=event_file,
            result_dir=result_dir,
            num_types=1,
            cut=frequencies[num],
            freq_range=freq_range,
            tmin=tmin,
            tmax=tmax,
        )

        times_list = [times[1:] for times in times_list]
        avg_evoked_list_frequencies.append(avg_evoked_list[0])

        plot_eeg = PlotEEG(
            channels=channels,
            result_dir=result_dir,
            is_show=False,
            is_save=True,
            eeg=eeg,
            eeg_times=eeg_times,
            eeg_filename=f"{num}_eeg_raw",
        )
        plot_eeg.plot_eeg()
        plot_eeg.plot_electrode(
            avg_evoked_list[0],
            times_list[0],
            filename=f"{num}_electrode",
        )

    avg_fp1_df = analyze_eeg.analyze_ssvep(
        fft_filename=fp1_file_path,
        frequencies=frequencies,
        freq_range=freq_range,
        result_dir=result_dir,
        early_cut=early_cut,
        scaled=scaled,
    )
    avg_fp2_df = analyze_eeg.analyze_ssvep(
        fft_filename=fp2_file_path,
        frequencies=frequencies,
        freq_range=freq_range,
        result_dir=result_dir,
        early_cut=early_cut,
        scaled=scaled,
    )
    
    plot_ssvep(
        df=avg_fp1_df,
        save_path=f"{result_dir}/fp1_ssvep.png",
    )
    plot_ssvep(
        df=avg_fp2_df,
        save_path=f"{result_dir}/fp2_ssvep.png",
    )

    recommend_selection(
        avg_evoked_list=avg_evoked_list,
        times_list=times_list,
        channels=channels,
        image_folder=image_folder,
    )


if __name__ == "__main__":
    import argparse
    import ast

    def parse_list(string: str):
        try:
            parsed_list = ast.literal_eval(string)
            if isinstance(parsed_list, list):
                return parsed_list
            else:
                raise argparse.ArgumentTypeError("Invalid list format")
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError("Invalid list format")

    parser = argparse.ArgumentParser(
        description="Insert arguments for function of erp ssvep select"
    )
    parser.add_argument(
        "--screen_width",
        type=int,
        default=1920,
        help="Set screen width of select task",
    )
    parser.add_argument(
        "--screen_height",
        type=int,
        default=1080,
        help="Set screen height of select task",
    )
    parser.add_argument(
        "--fs", type=int, default=256, help="Get resolution of EEG device"
    )
    parser.add_argument(
        "--channels",
        type=parse_list,
        default="['EEG_Fp1', 'EEG_Fp2']",
        help="Get channels of EEG device",
    )
    parser.add_argument(
        "--isi",
        type=int,
        default=1000,
        help="Set inter-stimulus interval of select task",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./images",
        help="Get image data path to use in the task",
    )
    parser.add_argument(
        "--frequencies",
        type=parse_list,
        default="[27.0, 13.0, 19.0, 23.0]",
        help="Get frequencies to use in the task",
    )
    parser.add_argument(
        "--experiment_duration",
        type=int,
        default=20,
        help="Set experiment_duration to use in the task",
    )
    parser.add_argument(
        "--event_save_path",
        type=str,
        default="./event",
        help="Set a record of events file saving path",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./plot",
        help="Set a EEG, ERP, SSVEP plots saving path",
    )
    parser.add_argument(
        "--result_dir_num",
        type=int,
        default=0,
        help="Set a EEG, ERP, SSVEP plots detailed saving path",
    )
    parser.add_argument(
        "--freq_range",
        type=float,
        default=5.0,
        help="Set freq_range to get ERP",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="Set epoch tmin to get ERP",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=1.0,
        help="Set epoch tmax to get ERP",
    )
    parser.add_argument(
        "--early_cut",
        type=int,
        default=0,
        help="Set an early cutting point of SSVEP",
    )
    parser.add_argument(
        "--scaled",
        type=bool,
        default=True,
        help="Set a scaling option of SSVEP",
    )
    args = parser.parse_args()

    erp_ssvep_select(
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        fs=args.fs,
        channels=args.channels,
        isi=args.isi,
        image_folder=f"{args.image_path}/gold",
        frequencies=args.frequencies,
        experiment_duration=args.experiment_duration,
        event_save_path=f"{args.event_save_path}",
        result_dir=f"{args.result_dir}/gold/{args.result_dir_num}",
        freq_range=args.freq_range,
        tmin=args.tmin,
        tmax=args.tmax,
        early_cut=args.early_cut,
        scaled=args.scaled,
    )
