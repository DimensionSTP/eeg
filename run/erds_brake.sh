screen_width=1920
screen_height=1080
fs=256
isi=7000
obstacle_playing_time=1500
image_path="./images"
background_order=1
num_trials=8
num_images=4
event_save_path="./event"
result_dir="./plot"
result_dir_num=0


python -u erds_brake.py \
    --screen_width=${screen_width} \
    --screen_height=${screen_height} \
    --fs=${fs} \
    --isi=${isi} \
    --obstacle_playing_time=${obstacle_playing_time} \
    --image_path=${image_path} \
    --background_order=${background_order} \
    --num_trials=${num_trials} \
    --num_images=${num_images} \
    --event_save_path=${event_save_path} \
    --result_dir=${result_dir} \
    --result_dir_num=${result_dir_num}