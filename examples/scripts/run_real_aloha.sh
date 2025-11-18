#!/bin/bash
proj_name=DSRL_pi05_Aloha
device_id=0

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Fill in remote host and port that serve the pi0 policy
export remote_host=""
export remote_port=""

python3 examples/launch_train_real_aloha.py \
--algorithm pixel_sac \
--env aloha \
--prefix dsrl_pi05_real \
--wandb_project ${proj_name} \
--batch_size 64 \
--discount 0.99 \
--seed 0 \
--max_steps 500000 \
--eval_interval 2000 \
--log_interval 100 \
--multi_grad_step 30 \
--resize_image 224 \
--query_freq 25 \
--hidden_dims 1024 1024 1024 \
--num_qs 2 \
--real_env_max_steps 1000 \
--external_camera high \
--wrist_camera left_wrist \
--gripper_indices -2 -1
