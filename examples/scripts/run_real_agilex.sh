#!/bin/bash
proj_name=DSRL_pi05_AgileX
device_id=0
camera_spec='{camera0: {type: orbbec, index_or_path: CP02653000ZL, width: 640, height: 480, fps: 30},camera1: {type: orbbec, index_or_path: CP02653000YJ, width: 640, height: 480, fps: 30},camera2: {type: orbbec, index_or_path: CP02653000YR, width: 640, height: 480, fps: 30},camera3: {type: orbbec, index_or_path: CP02653000Y4, width: 640, height: 480, fps: 30}}'

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 -m examples.launch_train_real_aloha \
--algorithm pixel_sac \
--env agilex \
--robot_type agilex \
--prefix dsrl_pi05_agilex \
--wandb_project ${proj_name} \
--batch_size 256 \
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
--image_order camera0 camera1 camera2 camera3 \
--num_cameras 4 \
--proprio_dim 7 \
--gripper_indices -1 \
--control_hz 30 \
--task "pick up the PCB board from the conveyor belt." \
--policy_checkpoint /home/test/jemotor/jemodel/pi05/1114_pi05_test/80000/ \
--policy_config pi05_agileX \
--agilex_port can_right \
--agilex_cameras_inline "${camera_spec}"
