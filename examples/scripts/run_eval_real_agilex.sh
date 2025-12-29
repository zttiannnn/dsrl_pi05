#!/bin/bash
proj_name=DSRL_pi05_AgileX_Eval
device_id=0
camera_spec='{camera0: {type: orbbec, index_or_path: CP02653000ZL, width: 640, height: 480, fps: 30},camera1: {type: orbbec, index_or_path: CP02653000YJ, width: 640, height: 480, fps: 30},camera2: {type: orbbec, index_or_path: CP02653000YR, width: 640, height: 480, fps: 30},camera3: {type: orbbec, index_or_path: CP02653000R4, width: 640, height: 480, fps: 30}}'

# Path to the trained RL agent checkpoint
RESTORE_PATH="/home/test/jemotor/dsrl_pi05/logs/DSRL_pi05_AgileX/dsrl_pi05_agilex_2025_12_02_10_23_06_0000--s-0/checkpoint6000"

export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Ensure PYTHONPATH includes openpi directories
export PYTHONPATH="${PYTHONPATH}:/home/test/jemotor/dsrl_pi05/openpi:/home/test/jemotor/dsrl_pi05/openpi/src"

python3 -m examples.eval_real_agilex \
--env agilex \
--robot_type agilex \
--resize_image 224 \
--query_freq 25 \
--hidden_dims 1024 1024 1024 \
--num_qs 2 \
--real_env_max_steps 10000 \
--image_order camera0 camera1 camera2 camera3 \
--num_cameras 4 \
--proprio_dim 7 \
--img_feature_dim 2048 \
--gripper_indices -1 \
--control_hz 30 \
--task "Pick up the PCB board from the conveyor belt and place it into the yellow container." \
--policy_checkpoint /home/test/jemotor/jemodel/pi05/1114_pi05_test/50000/ \
--policy_config pi05_agileX \
--agilex_port can_right \
--agilex_cameras_inline "${camera_spec}" \
--restore_path "${RESTORE_PATH}" \
--eval_episodes 10 \
--no_rtc
