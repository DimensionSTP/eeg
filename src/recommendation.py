from typing import List

from PIL import Image, ImageDraw, ImageFont
import pandas as pd


def recommend_combination(
    avg_evoked_list: List, times_list: List, channels: List, clothes_type: str
):
    max_values_per_channels = []
    for channel_idx in range(len(channels)):
        max_values = []
        for time in range(len(times_list)):
            selected_indices = [
                index
                for index, value in enumerate(times_list[time])
                if 0.1 <= value <= 0.5
            ]
            start_index = selected_indices[0]
            end_index = selected_indices[-1]

            max_value = max(
                avg_evoked_list[time][channel_idx][start_index : end_index + 1]
            )
            max_values.append(max_value)
        max_values_per_channels.append(max_values)

    indices_of_largest_values_per_channels = []
    for channel in range(len(max_values_per_channels)):
        indices_of_largest_values = sorted(
            range(len(max_values_per_channels[channel])),
            key=lambda i: max_values_per_channels[channel][i],
            reverse=True,
        )[:3]
        largest_values = [
            max_values_per_channels[channel][i] for i in indices_of_largest_values
        ]
        top_values_and_indices = [
            (value, index)
            for value, index in zip(largest_values, indices_of_largest_values)
        ]
        indices_of_largest_values_per_channels.append(top_values_and_indices)

    top_values_and_indices = sum(indices_of_largest_values_per_channels, [])
    sorted_top_values_and_indices = sorted(
        top_values_and_indices, key=lambda i: i[0], reverse=True
    )
    top_recommendations = []
    seen_indices = set()
    for t in sorted_top_values_and_indices:
        if t[1] not in seen_indices and len(top_recommendations) < 3:
            top_recommendations.append(t)
            seen_indices.add(t[1])
    top_indices = [t[1] + 1 for t in top_recommendations]
    if clothes_type == "bottoms":
        for index in top_indices:
            print(f"당신이 끌리는 하의 조합은 {index}번 하의입니다.")
            image_filename = f"./images/bottoms/B{index}.jpg"
            image = Image.open(image_filename)
            image.show()
    elif clothes_type == "shoes":
        for index in top_indices:
            print(f"당신이 끌리는 신발의 조합은 {index}번 신발입니다.")
            image_filename = f"./images/shoes/S{index}.jpg"
            image = Image.open(image_filename)
            image.show()
    else:
        raise ValueError("Invalid clothes type")


def recommend_answer_ssvep(
    fp1_df:pd.DataFrame, fp2_df:pd.DataFrame, screen_width: int, screen_height: int, frequencies: List, image_folder: str, correct_num: int, result_dir: str
):
    combined_df = pd.concat([fp1_df, fp2_df], axis=1)
    max_values = combined_df.max()
    max_column_name = max_values.idxmax()
    for i in range(len(frequencies)):
        if frequencies[i] == int(float(max_column_name[:-2])):
            image_num = i*2+1
            image = Image.open(f"{image_folder}/{image_num}.png")
            image = image.resize((screen_width, screen_height))
            draw = ImageDraw.Draw(image)
            if i == correct_num-1:
                text = "정답입니다!"
            else:
                text = f"틀렸습니다. 정답은 {correct_num}번 입니다."
            font_size = 50
            font = ImageFont.truetype("C:/Windows/Fonts/batang.ttc", font_size)
            
            # 텍스트 너비와 높이를 구하고 이미지 중앙 상단에 위치시키기
            text_width, text_height = draw.textsize(text, font=font)
            text_x = (image.width - text_width) // 2
            text_y = 10  # 상단과 적당한 간격 두기
            
            # 텍스트 그리기 (하얀색)
            draw.text((text_x, text_y), text, font=font, fill="white")
            
            # 변경된 이미지 저장
            image.save(f"{result_dir}/answer.png")
            image.show(f"{result_dir}/answer.png")