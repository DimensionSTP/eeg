from typing import List

from PIL import Image


def recommend(
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
