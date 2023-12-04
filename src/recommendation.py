from typing import List

from PIL import Image, ImageDraw, ImageFont
import pandas as pd


def recommend_combination(
    avg_evoked_list: List, times_list: List, channels: List, image_folder: str, clothes_type: str,
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
            image_filename = f"{image_folder}/B{index}.jpg"
            image = Image.open(image_filename)
            image.show()
    elif clothes_type == "shoes":
        for index in top_indices:
            print(f"당신이 끌리는 신발의 조합은 {index}번 신발입니다.")
            image_filename = f"{image_folder}/S{index}.jpg"
            image = Image.open(image_filename)
            image.show()
    else:
        raise ValueError("Invalid clothes type")


def recommend_celebrity(
    avg_evoked_list: List, times_list: List, channels: List, sex: str, image_folder: str, result_dir: str, screen_width: int, screen_height: int,
):
    male_celebrities = [
        "공유",
        "송중기",
        "박서준",
        "유연석",
        "이종석",
        "김선호",
        "정해인",
        "이제훈",
        "이동욱",
        "뷔",
        "차은우"
    ]
    female_celebrities = [
        "김유정",
        "전소미",
        "한소희",
        "수지",
        "안유진",
        "카리나",
        "윈터",
        "미연",
        "태연",
        "아이유",
        "윤아"
    ]
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
    top_index = sorted_top_values_and_indices[0][1]
    erp_fp1_path = f"{result_dir}/{sex}_{top_index+1}_electrode_average_EEG_Fp1.png"
    erp_fp2_path = f"{result_dir}/{sex}_{top_index+1}_electrode_average_EEG_Fp2.png"
    # Let"s start fresh and simply vertically stack the 2nd and 3rd images as requested by the user.

    # Reload the EEG images
    erp_fp1_plot = Image.open(erp_fp1_path)
    erp_fp2_plot = Image.open(erp_fp2_path)

    # Create a new image with the combined height of the two EEG images and the width of the widest one
    vertical_combined_height = erp_fp1_plot.height + erp_fp2_plot.height
    vertical_combined_width = max(erp_fp1_plot.width, erp_fp2_plot.width)

    # Create a new image with a black background
    vertical_combined_image = Image.new("RGB", (vertical_combined_width, vertical_combined_height), "black")

    # Paste the first EEG image at the top
    vertical_combined_image.paste(erp_fp1_plot, (0, 0))

    # Paste the second EEG image directly below the first
    vertical_combined_image.paste(erp_fp2_plot, (0, erp_fp1_plot.height))

    # Save the vertically combined image
    vertical_combined_image_path = f"{result_dir}/erp_combined.png"
    vertical_combined_image.save(vertical_combined_image_path)

    # Load images
    if sex == "males":
        celebrity_path = f"{image_folder}/M{top_index+1}.jpg"
        text = f"당신이 끌리는 연예인은 {male_celebrities[top_index]}입니다."
    elif sex == "females":
        celebrity_path = f"{image_folder}/F{top_index+1}.jpg"
        text = f"당신이 끌리는 연예인은 {female_celebrities[top_index]}입니다."
    else:
        raise ValueError("Invalid sex")
    erp_combined_path = f"{result_dir}/erp_combined.png"
    celebrity_image = Image.open(celebrity_path)
    erp_combined_plot = Image.open(erp_combined_path)

    # Determine the new width and height for the combined image
    new_width = max(celebrity_image.width + erp_combined_plot.width, screen_width)
    new_height = max(celebrity_image.height, erp_combined_plot.height, screen_height)

    # Create a new image with the determined dimensions and black background
    combined_image = Image.new("RGB", (new_width, new_height), color="black")

    # Calculate the position for the first image (left)
    x_offset = int((new_width - (celebrity_image.width + erp_combined_plot.width)) / 2)
    y_offset = int((new_height - celebrity_image.height) / 2)

    # Paste the first image onto the combined image
    combined_image.paste(celebrity_image, (x_offset, y_offset))

    # Calculate the position for the second image (right)
    x_offset += celebrity_image.width

    # Paste the second image onto the combined image
    combined_image.paste(erp_combined_plot, (x_offset, y_offset))

    # Add text to the combined image
    draw = ImageDraw.Draw(combined_image)
    font_size = 50  # Starting font size
    font_path = "C:/Windows/Fonts/batang.ttc"
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the width and height of the text to be added
    text_width, text_height = draw.textsize(text, font=font)

    # While the text width is too large, reduce the font size
    while text_width > new_width and font_size > 10:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)

    # Calculate the position for the text
    text_x = (new_width - text_width) / 2
    text_y = 10  # A small padding from the top

    # Draw the text onto the combined image
    draw.text((text_x, text_y), text, font=font, fill="white")

    # Save the combined image
    combined_image_path = f"{result_dir}/recommendation.png"
    combined_image.save(combined_image_path)
    combined_image.show()


def recommend_answer_ssvep(
    fp1_df:pd.DataFrame, fp2_df:pd.DataFrame, screen_width: int, screen_height: int, frequencies: List, image_folder: str, correct_num: int, result_dir: str,
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


def recommend_selection(
    avg_evoked_list: List, times_list: List, channels: List, image_folder: str,
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
    top_index = sorted_top_values_and_indices[0][1]
    print(f"your selection is {top_index*2+1}")
    image_filename = f"{image_folder}/{top_index*2+1}.png"
    image = Image.open(image_filename)
    image.show()