import datetime
import time
import csv

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
    time.sleep(30)
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