import datetime
import time
import csv
from typing import List

import pygame
import cv2


def combination_task(
    screen_width: int,
    screen_height: int,
    isi: int,
    top_image_path: str,
    image_folder: str,
    num_trials: int,
    num_images: int,
    event_save_path: str,
    clothes_type: str,
):
    pygame.init()

    screen = pygame.display.set_mode((screen_width, screen_height))
    current_time = datetime.datetime.now()
    hour = str(current_time).split(" ")[1].split(":")[0]
    min = str(current_time).split(" ")[1].split(":")[1]
    sec = str(current_time).split(" ")[1].split(":")[2]

    filename = f"{event_save_path}/combination_event_{hour}.{min}.{sec}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ISI", "RT", "Response", "Stimulus"])

    for _ in range(num_trials):
        for num_image in range(num_images):
            top_image = pygame.image.load(top_image_path)
            screen.blit(
                top_image,
                (
                    screen_width // 2 - top_image.get_width() // 2,
                    screen_height // 2 - top_image.get_height() // 2,
                ),
            )
            pygame.display.flip()

            time.sleep(isi / 1000.0)

            start_time = pygame.time.get_ticks()

            if clothes_type == "bottoms":
                task_image = pygame.image.load(f"{image_folder}/B{num_image+1}.jpg")
            elif clothes_type == "shoes":
                task_image = pygame.image.load(f"{image_folder}/S{num_image+1}.jpg")
            else:
                raise ValueError("Invalid clothes type")
            screen.blit(
                task_image,
                (
                    screen_width // 2 - task_image.get_width() // 2,
                    screen_height // 2 - task_image.get_height() // 2,
                ),
            )
            pygame.display.flip()

            response = "CR"
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        response = "HIT"
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            end_time = pygame.time.get_ticks()

            time.sleep(isi / 1000.0)

            if response == "HIT":
                rt = end_time - start_time
            else:
                rt = 1000

            # CSV 파일에 결과 기록
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        isi,
                        rt,
                        response,
                        num_image + 1,
                    ]
                )
    time.sleep(10)
    pygame.quit()


def celebrity_task(
    screen_width: int,
    screen_height: int,
    isi: int,
    background_path: str,
    image_folder: str,
    num_trials: int,
    num_images: int,
    event_save_path: str,
    sex: str,
):
    pygame.init()

    screen = pygame.display.set_mode((screen_width, screen_height))
    current_time = datetime.datetime.now()
    hour = str(current_time).split(" ")[1].split(":")[0]
    min = str(current_time).split(" ")[1].split(":")[1]
    sec = str(current_time).split(" ")[1].split(":")[2]

    filename = f"{event_save_path}/celebrity_event_{hour}.{min}.{sec}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ISI", "RT", "Response", "Stimulus"])

    for _ in range(num_trials):
        for num_image in range(num_images):
            top_image = pygame.image.load(background_path)
            screen.blit(
                top_image,
                (
                    screen_width // 2 - top_image.get_width() // 2,
                    screen_height // 2 - top_image.get_height() // 2,
                ),
            )
            pygame.display.flip()

            time.sleep(isi / 1000.0)

            start_time = pygame.time.get_ticks()

            if sex == "males":
                task_image = pygame.image.load(f"{image_folder}/M{num_image+1}.jpg")
            elif sex == "females":
                task_image = pygame.image.load(f"{image_folder}/F{num_image+1}.jpg")
            else:
                raise ValueError("Invalid sex type")
            screen.blit(
                task_image,
                (
                    screen_width // 2 - task_image.get_width() // 2,
                    screen_height // 2 - task_image.get_height() // 2,
                ),
            )
            pygame.display.flip()

            response = "CR"
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        response = "HIT"
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            end_time = pygame.time.get_ticks()

            time.sleep(isi / 1000.0)

            if response == "HIT":
                rt = end_time - start_time
            else:
                rt = 1000

            # CSV 파일에 결과 기록
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        isi,
                        rt,
                        response,
                        num_image + 1,
                    ]
                )
    time.sleep(10)
    pygame.quit()


def brake_task(
    screen_width: int,
    screen_height: int,
    isi: int,
    obstacle_playing_time: int,
    background_path: str,
    image_folder: str,
    num_trials: int,
    num_images: int,
    event_save_path: str,
):
    pygame.init()

    # 실험 데이터 초기화 및 실험 시작 시간 기록
    current_time = datetime.datetime.now()
    hour = str(current_time).split(" ")[1].split(":")[0]
    min = str(current_time).split(" ")[1].split(":")[1]
    sec = str(current_time).split(" ")[1].split(":")[2]

    filename = f"{event_save_path}/brake_event_{hour}.{min}.{sec}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ISI", "RT", "Response", "Stimulus"])

    # event_data = []
    for _ in range(num_trials):
        cap = cv2.VideoCapture(background_path)
        # 동영상 재생 및 장애물 이미지 표시
        for num_image in range(num_images):
            obstacle = cv2.imread(f"{image_folder}/O{num_image+1}.png", cv2.IMREAD_UNCHANGED)
            # 동영상 재생
            start_video_time = time.time()
            while time.time() - start_video_time < int(isi / 1000.0):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (screen_width, screen_height))
                cv2.imshow("Video", frame)
                if cv2.waitKey(28) & 0xFF == ord("q"):
                    break

            # 장애물 이미지 표시
            start_obstacle_time = time.time()
            while time.time() - start_obstacle_time < float(obstacle_playing_time / 1000.0):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (screen_width, screen_height))

                # 장애물 이미지 크기와 위치 계산
                oh, ow = obstacle.shape[:2]
                y, x = (frame.shape[0] - oh) // 2, (frame.shape[1] - ow) // 2

                # 장애물 이미지 오버레이
                alpha_s = obstacle[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    frame[y:y+oh, x:x+ow, c] = (alpha_s * obstacle[:, :, c] + alpha_l * frame[y:y+oh, x:x+ow, c])

                cv2.imshow("Video", frame)
                if cv2.waitKey(28) & 0xFF == ord("q"):
                    break

            # 사용자 입력 기록
            response, rt = "CR", 2500
            start_input_time = time.time()
            while time.time() - start_input_time < float(obstacle_playing_time / 1000.0):
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        rt = int((time.time() - start_input_time) * 1000)
                        response = "HIT"
                        break
                if response == "HIT":
                    break
                        # CSV 파일에 결과 기록
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        isi,
                        rt,
                        response,
                        num_image + 1,
                    ]
                )    

    # 동영상 및 Pygame 종료
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    time.sleep(10)


def grap_task(
    screen_width: int,
    screen_height: int,
    isi: int,
    obstacle_playing_time: int,
    background_path: str,
    image_folder: str,
    num_trials: int,
    num_images: int,
    event_save_path: str,
):
    pygame.init()

    # 실험 데이터 초기화 및 실험 시작 시간 기록
    current_time = datetime.datetime.now()
    hour = str(current_time).split(" ")[1].split(":")[0]
    min = str(current_time).split(" ")[1].split(":")[1]
    sec = str(current_time).split(" ")[1].split(":")[2]

    filename = f"{event_save_path}/grap_event_{hour}.{min}.{sec}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ISI", "RT", "Response", "Stimulus"])

    # event_data = []
    for _ in range(num_trials):
        cap = cv2.VideoCapture(background_path)
        # 동영상 재생 및 장애물 이미지 표시
        for num_image in range(num_images):
            obstacle = cv2.imread(f"{image_folder}/O{num_image+1}.png", cv2.IMREAD_UNCHANGED)
            # 동영상 재생
            start_video_time = time.time()
            while time.time() - start_video_time < int(isi / 1000.0):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (screen_width, screen_height))
                cv2.imshow("Video", frame)
                if cv2.waitKey(28) & 0xFF == ord("q"):
                    break

            # 장애물 이미지 표시
            start_obstacle_time = time.time()
            while time.time() - start_obstacle_time < float(obstacle_playing_time / 1000.0):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (screen_width, screen_height))

                # 장애물 이미지 크기와 위치 계산
                oh, ow = obstacle.shape[:2]
                y, x = (frame.shape[0] - oh) // 2, (frame.shape[1] - ow) // 2

                # 장애물 이미지 오버레이
                alpha_s = obstacle[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    frame[y:y+oh, x:x+ow, c] = (alpha_s * obstacle[:, :, c] + alpha_l * frame[y:y+oh, x:x+ow, c])

                cv2.imshow("Video", frame)
                if cv2.waitKey(28) & 0xFF == ord("q"):
                    break

            # 사용자 입력 기록
            response, rt = "CR", 2500
            start_input_time = time.time()
            while time.time() - start_input_time < float(obstacle_playing_time / 1000.0):
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        rt = int((time.time() - start_input_time) * 1000)
                        response = "HIT"
                        break
                if response == "HIT":
                    break
                        # CSV 파일에 결과 기록
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        isi,
                        rt,
                        response,
                        0,
                    ]
                )    

    # 동영상 및 Pygame 종료
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    time.sleep(10)


def quiz_task(
    screen_width: int,
    screen_height: int,
    image_folder: str,
    frequencies: List,
    experiment_duration: int,
):
    # Pygame 초기화
    pygame.init()

    # 화면 크기 및 설정
    screen_width = screen_width
    screen_height = screen_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("SSVEP Experiment")

    # 이미지 로딩
    images = [pygame.image.load(f"{image_folder}/{i}.png") for i in range(1, 9)]

    # 이미지 위치 및 크기 계산
    image_width, image_height = images[0].get_size()
    image_positions = [
        ((screen_width / 4) - (image_width / 2), (screen_height / 4) - (image_height / 2)),  # 상단 왼쪽
        ((screen_width * 3 / 4) - (image_width / 2), (screen_height / 4) - (image_height / 2)),  # 상단 오른쪽
        ((screen_width / 4) - (image_width / 2), (screen_height * 3 / 4) - (image_height / 2)),  # 하단 왼쪽
        ((screen_width * 3 / 4) - (image_width / 2), (screen_height * 3 / 4) - (image_height / 2))  # 하단 오른쪽
    ]

    # 주파수에 따른 이미지 변경 주기 계산
    change_intervals = [1 / f for f in frequencies]

    # 타이머 및 이미지 인덱스 초기화
    timers = [time.time()] * 4
    current_images = [0, 2, 4, 6]  # 각 화면별 첫 이미지 인덱스

    # 실험 시작 시간
    experiment_start_time = time.time()

    # 메인 루프
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 실험 시간이 지나면 종료
        if time.time() - experiment_start_time >= experiment_duration:
            running = False

        # 각 화면별 이미지 변경
        for i in range(4):
            if time.time() - timers[i] >= change_intervals[i]:
                current_images[i] = (current_images[i] + 1) % 2 + i * 2  # 각 화면별로 다음 이미지 선택
                timers[i] = time.time()

        # 화면 지우기
        screen.fill((0, 0, 0))

        # 각 화면에 이미지 표시
        for i in range(4):
            rect = images[current_images[i]].get_rect()
            rect.topleft = image_positions[i]
            screen.blit(images[current_images[i]], rect)

        # 중앙 고정점 그리기 (예: 하얀색 십자가)
        cross_center = (screen_width / 2, screen_height / 2)
        pygame.draw.line(screen, (255, 255, 255), (cross_center[0] - 10, cross_center[1]), (cross_center[0] + 10, cross_center[1]), 2)
        pygame.draw.line(screen, (255, 255, 255), (cross_center[0], cross_center[1] - 10), (cross_center[0], cross_center[1] + 10), 2)

        pygame.display.flip()

    # Pygame 종료
    pygame.quit()
    time.sleep(10)


def selection_task(
    screen_width: int,
    screen_height: int,
    isi: int,
    image_folder: str,
    frequencies: List,
    experiment_duration: int,
    event_save_path: str,
):
    # Pygame 초기화
    pygame.init()

    # 화면 크기 및 설정
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Selection Experiment")
    current_time = datetime.datetime.now()
    hour = str(current_time).split(" ")[1].split(":")[0]
    min = str(current_time).split(" ")[1].split(":")[1]
    sec = str(current_time).split(" ")[1].split(":")[2]

    filename = f"{event_save_path}/select_event_{hour}.{min}.{sec}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ISI", "RT", "Stimulus"])

    # 이미지 로딩
    images = [pygame.image.load(f"{image_folder}/{i}.png") for i in range(1, 9)]

    # 이미지 위치 및 크기 계산
    image_width, image_height = images[0].get_size()
    image_positions = [
        ((screen_width / 4) - (image_width / 2), (screen_height / 4) - (image_height / 2)),  # 상단 왼쪽
        ((screen_width * 3 / 4) - (image_width / 2), (screen_height / 4) - (image_height / 2)),  # 상단 오른쪽
        ((screen_width / 4) - (image_width / 2), (screen_height * 3 / 4) - (image_height / 2)),  # 하단 왼쪽
        ((screen_width * 3 / 4) - (image_width / 2), (screen_height * 3 / 4) - (image_height / 2))  # 하단 오른쪽
    ]

    # 주파수에 따른 깜박임 주기 계산 (1초 동안 깜박이는 시간)
    flash_intervals = [1 / f for f in frequencies]  # 주파수의 반주기

    # 타이머 및 상태 초기화
    timers = [0] * len(frequencies)
    flash_states = [False] * len(frequencies)  # 깜박임 상태 (True: 깜박임, False: 검은 화면)

    # 실험 시작 시간
    experiment_start_time = time.time()

    # 메인 루프
    running = True
    while running:
        current_time = time.time()

        # 실험 시간이 지나면 종료
        if current_time - experiment_start_time >= experiment_duration:
            running = False

        # 화면 지우기
        screen.fill((0, 0, 0))
        

        # 각 화면별 깜박임 및 표시
        for i in range(4):
            if current_time - timers[i] >= isi / 1000.0:
                # 1초 후 깜박임 상태 전환
                flash_states[i] = not flash_states[i]
                timers[i] = current_time

            if flash_states[i]:
                # 깜박임 상태일 때만 이미지 표시
                time_since_flash_start = current_time - timers[i]
                if time_since_flash_start % flash_intervals[i] < flash_intervals[i] / 2:
                    rect = images[i * 2].get_rect()
                    rect.topleft = image_positions[i]
                    screen.blit(images[i * 2], rect)

        # 중앙 고정점 그리기 (예: 하얀색 십자가)
        cross_center = (screen_width / 2, screen_height / 2)
        pygame.draw.line(screen, (255, 255, 255), (cross_center[0] - 10, cross_center[1]), (cross_center[0] + 10, cross_center[1]), 2)
        pygame.draw.line(screen, (255, 255, 255), (cross_center[0], cross_center[1] - 10), (cross_center[0], cross_center[1] + 10), 2)

        pygame.display.flip()

        # CSV 파일에 결과 기록
        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    isi,
                    isi,
                    1,
                ]
            )
            
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Pygame 종료
    time.sleep(10)
    pygame.quit()