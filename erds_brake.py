import os
from typing import List
from datetime import datetime

import pandas as pd

from src.task import brake_task
from src.analysis import AnalyzeEEG
from src.plot import PlotEEG


def erds_brake(
    screen_width: int,
    screen_height: int,
    fs: int,
    channels: List,
    isi: int,
    obstacle_playing_time: int,
    background_path: str,
    image_folder: str,
    num_trials: int,
    num_images: int,
    event_save_path: str,
    result_dir: str,
    erd_lowcut: float = 8.0,
    erd_highcut: float = 11.0,
    ers_lowcut: float = 26.0,
    ers_highcut: float = 30.0,
    tmin: float = -4.0,
    tmax: float = 4.0,
):
    today = str(datetime.now().date())
    if not os.path.exists(f"./data/{today}"):
        os.makedirs(f"./data/{today}")
    if not os.path.exists(f"./event/{today}"):
        os.makedirs(f"./event/{today}")

    brake_task(
        screen_width=screen_width,
        screen_height=screen_height,
        isi=isi,
        obstacle_playing_time=obstacle_playing_time,
        background_path=background_path,
        image_folder=image_folder,
        num_trials=num_trials,
        num_images=num_images,
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

    analyze_eeg = AnalyzeEEG(channels=channels, fs=fs)
    eeg, eeg_times, erd_avg_evoked_list, erd_times_list, ers_avg_evoked_list, ers_times_list = analyze_eeg.analyze_erds(
        eeg_filename=data_file_path,
        event_filename=event_file,
        result_dir=result_dir,
        num_types=num_images,
        erd_lowcut=erd_lowcut,
        erd_highcut=erd_highcut,
        ers_lowcut=ers_lowcut,
        ers_highcut=ers_highcut,
        tmin=tmin,
        tmax=tmax,
    )

    plot_eeg = PlotEEG(
        channels=channels,
        result_dir=result_dir,
        is_show=False,
        is_save=True,
        eeg=eeg,
        eeg_times=eeg_times,
        eeg_filename="eeg_raw",
    )
    plot_eeg.plot_eeg()
    for i in range(num_images):
        plot_eeg.plot_electrode(
            erd_avg_evoked_list[i],
            erd_times_list[i],
            filename=f"brake_erd_{i+1}_electrode",
        )
    for i in range(num_images):
        plot_eeg.plot_electrode(
            ers_avg_evoked_list[i],
            ers_times_list[i],
            filename=f"brake_ers_{i+1}_electrode",
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
        description="Insert arguments for function of erds brake"
    )
    parser.add_argument(
        "--screen_width",
        type=int,
        default=1920,
        help="Set screen width of brake task",
    )
    parser.add_argument(
        "--screen_height",
        type=int,
        default=1080,
        help="Set screen height of brake task",
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
        default=7000,
        help="Set inter-stimulus interval of brake task",
    )
    parser.add_argument(
        "--obstacle_playing_time",
        type=int,
        default=1500,
        help="Set obstacle playing time of brake task",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./images",
        help="Get image data path to use in the task",
    )
    parser.add_argument(
        "--backgrounds_order",
        type=int,
        default=1,
        help="Set order of upper clothes to use in the task",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=8,
        help="Set number of trials to use in the task",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Set number of clothes to use in the task",
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
        help="Set a EEG, ERDS plots saving path",
    )
    parser.add_argument(
        "--result_dir_num",
        type=int,
        default=0,
        help="Set a EEG, ERDS plots detailed saving path",
    )
    parser.add_argument(
        "--erd_lowcut",
        type=float,
        default=8.0,
        help="Set butter filter lowcut to get ERD",
    )
    parser.add_argument(
        "--erd_highcut",
        type=float,
        default=11.0,
        help="Set butter filter highcut to get ERD",
    )
    parser.add_argument(
        "--ers_lowcut",
        type=float,
        default=26.0,
        help="Set butter filter lowcut to get ERS",
    )
    parser.add_argument(
        "--ers_highcut",
        type=float,
        default=30.0,
        help="Set butter filter highcut to get ERS",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=-4.0,
        help="Set epoch tmin to get ERDS",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=4.0,
        help="Set epoch tmax to get ERDS",
    )
    args = parser.parse_args()

    erds_brake(
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        fs=args.fs,
        channels=args.channels,
        isi=args.isi,
        obstacle_playing_time=args.obstacle_playing_time,
        background_path=f"{args.image_path}/backgrounds/B{args.backgrounds_order}.mp4",
        image_folder=f"{args.image_path}/obstacles",
        num_trials=args.num_trials,
        num_images=args.num_images,
        event_save_path=f"{args.event_save_path}",
        result_dir=f"{args.result_dir}/brake/{args.result_dir_num}",
        erd_lowcut=args.erd_lowcut,
        erd_highcut=args.erd_highcut,
        ers_lowcut=args.ers_lowcut,
        ers_highcut=args.ers_highcut,
        tmin=args.tmin,
        tmax=args.tmax,
    )