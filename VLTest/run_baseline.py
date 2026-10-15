import os

os.system(
    "CUDA_VISIBLE_DEVICES=0 python perturber_baseline.py \
    --output_path=outputs\
    --task=itm \
    --seed_nums=35 \
    --pre_seed_num=10 \
    --pre_level_start=0\
    --pre_level_step=0.0005 \
    --pre_time=400 \
    --seed 123456"
)
