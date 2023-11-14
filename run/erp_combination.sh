screen_width=1920
screen_height=1080
fs=256
isi=1000
image_path="./images"
tops_order=1
clothes_type="bottoms"
num_trials=10
num_images=30
event_save_path="./event"
result_dir="./plot"
result_dir_num=0


python -u erp_combination.py \
    --screen_width=${screen_width} \
    --screen_height=${screen_height} \
    --fs=${fs} \
    --isi=${isi} \
    --image_path=${image_path} \
    --tops_order=${tops_order} \
    --clothes_type=${clothes_type} \
    --num_trials=${num_trials} \
    --num_images=${num_images} \
    --event_save_path=${event_save_path} \
    --result_dir=${result_dir} \
    --result_dir_num=${result_dir_num}