import os
from typing import List
from datetime import datetime

import pandas as pd

from src.task import quiz_task
from src.analysis import AnalyzeEEG
from src.plot import plot_ssvep
from src.recommendation import recommend_answer_ssvep

def ssvep_quiz(
    screen_width: int,
    screen_height: int,
    fs: int,
    channels: List,
    image_folder: str,
    frequencies: List,
    experiment_duration: int,
    freq_range: float,
    correct_num: int,
    result_dir: str,
):
    today = str(datetime.now().date())
    if not os.path.exists(f"./data/{today}"):
        os.makedirs(f"./data/{today}")
    if not os.path.exists(f"./event/{today}"):
        os.makedirs(f"./event/{today}")

    quiz_task(
        screen_width=screen_width,
        screen_height=screen_height,
        image_folder=image_folder,
        frequencies=frequencies,
        experiment_duration=experiment_duration
    )

    rawdata_folders = os.listdir("C:/MAVE_RawData")

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
    avg_fp1_df = analyze_eeg.analyze_ssvep(
        fft_filename=fp1_file_path,
        frequencies=frequencies,
        freq_range=freq_range,
        result_dir=result_dir
    )
    avg_fp2_df = analyze_eeg.analyze_ssvep(
        fft_filename=fp2_file_path,
        frequencies=frequencies,
        freq_range=freq_range,
        result_dir=result_dir
    )
    
    plot_ssvep(
        df=avg_fp1_df,
        save_path=f"{result_dir}/fp1.png"
    )
    plot_ssvep(
        df=avg_fp2_df,
        save_path=f"{result_dir}/fp2.png"
    )
    
    recommend_answer_ssvep(
        fp1_df=avg_fp1_df, fp2_df=avg_fp2_df, screen_width=screen_width, screen_height=screen_height, frequencies=frequencies, image_folder=image_folder, correct_num=correct_num, result_dir=result_dir
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
        "--image_path",
        type=str,
        default="./images",
        help="Get image data path to use in the task",
    )
    parser.add_argument(
        "--frequencies",
        type=parse_list,
        default="[27, 13, 19, 23]",
        help="Get frequencies to use in the task",
    )
    parser.add_argument(
        "--experiment_duration",
        type=int,
        default=60,
        help="Get experiment duration to use in the task",
    )
    parser.add_argument(
        "--freq_range",
        type=float,
        default=0.4,
        help="Get frequency range to use in the task",
    )
    parser.add_argument(
        "--correct_num",
        type=int,
        default=2,
        help="Get number of correct answer to use in the task",
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
    args = parser.parse_args()

    ssvep_quiz(
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        fs=args.fs,
        channels=args.channels,
        image_folder=f"{args.image_path}/quiz",
        frequencies=args.frequencies,
        experiment_duration=args.experiment_duration,
        freq_range=args.freq_range,
        correct_num=args.correct_num,
        result_dir=f"{args.result_dir}/quiz/{args.result_dir_num}",
    )
