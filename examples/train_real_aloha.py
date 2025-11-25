#! /usr/bin/env python
"""Entry point for running pi05 / Aloha real-world training."""

import logging
import os
import tempfile
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Dict
from jax.experimental.compilation_cache import compilation_cache

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name

from examples.agilex_env_wrapper import AgileXFollowerEnv
from examples.aloha_env_wrapper import AlohaRobotEnv
from examples.local_policy_client import LocalPolicyClient
from examples.train_utils_real_aloha import trajwise_alternating_training_loop

home_dir = os.environ["HOME"]
compilation_cache.initialize_cache(os.path.join(home_dir, "jax_compilation_cache"))


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension."""

    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """Minimal observation space placeholder for constructing the learner."""

    def __init__(self, variant):
        self.variant = variant
        image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {"pixels": Box(low=0, high=255, shape=image_shape, dtype=np.uint8)}
        if variant.add_states:
            state_dim = variant.proprio_dim + variant.img_feature_dim
            obs_dict["state"] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32), dtype=np.float32)


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    logging.info("num devices %s", num_devices)
    logging.info("batch size %s", variant.batch_size)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = variant["train_kwargs"]
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = variant.max_steps

    if not variant.prefix:
        import uuid

        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    outputdir = os.path.join(os.environ["EXP"], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print("writing to output dir", outputdir)

    group_name = variant.prefix + "_" + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != "",
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=wandb_output_dir,
        group_name=group_name,
    )

    # ========== 1. 初始化本地策略客户端（子进程推理） ==========
    # LocalPolicyClient 在独立进程加载 pi05_agileX 策略，避免主进程 CUDA 内存冲突
    # 提供两个核心接口：
    #   - infer(obs, noise) → 返回动作序列用于环境交互
    #   - get_prefix_rep(obs) → 返回视觉特征用于 RL state 拼接
    checkpoint = getattr(variant, "policy_checkpoint", "")
    if not checkpoint:
        raise ValueError("--policy_checkpoint must point to a trained pi05_agileX checkpoint")
    cfg_name = getattr(variant, "policy_config", "pi05_agileX")
    default_prompt = variant.instruction  # 任务描述，传给策略作为条件输入
    agent_dp = LocalPolicyClient(checkpoint, config_name=cfg_name, default_prompt=default_prompt)
    metadata = agent_dp.get_server_metadata() or {}  # 可能包含 reset_pose 等配置
    logging.info(
        "Using LocalPolicyClient (config=%s, checkpoint=%s), metadata: %s",
        cfg_name,
        checkpoint,
        metadata,
    )

    # ========== 2. 初始化硬件环境（AgileX 或 Aloha） ==========
    # 根据 --robot_type 选择对应的 wrapper：
    #   - AgileX: 连接松灵机械臂 + YAML 配置的摄像头（OpenCV/Orbbec）
    #   - Aloha:  连接双臂 Aloha 系统（通过 openpi05 环境）
    # 环境提供 reset()、get_observation()、step(action) 接口供训练循环调用
    robot_type = getattr(variant, "robot_type", variant.env).lower()
    logging.info("Selected robot type: %s", robot_type)
    if robot_type == "agilex":
        if not variant.agilex_port:
            raise ValueError("--agilex_port must be provided when robot_type=agilex")
        env = AgileXFollowerEnv(
            port=variant.agilex_port,                # 串口设备路径，如 /dev/ttyACM0
            robot_id=getattr(variant, "agilex_robot_id", "left"),  # 机械臂 ID
            camera_config=getattr(variant, "agilex_camera_dict", None),
            camera_config_path=None,  # 摄像头 YAML 配置
            max_relative_target=getattr(variant, "agilex_max_relative_target", None),
            use_degrees=bool(getattr(variant, "agilex_use_degrees", False)),
            prompt=variant.instruction,  # 任务提示词（记录用，实际未在 env 中使用）
        )
        eval_env = env
        logging.info("Created AgileX follower environment")
    else:
        logging.info("initializing Aloha environment...")
        env = AlohaRobotEnv(render_size=variant.resize_image, reset_position=metadata.get("reset_pose"))
        eval_env = env
        logging.info("created the aloha env!")

    robot_config = dict(
        image_order=list(variant.image_order),  # 直接使用 camera0, camera1, camera2, camera3
        external_camera=variant.image_order[0], # 默认使用第一个相机作为外部相机用于视频录制
        max_timesteps=variant.real_env_max_steps,
        gripper_indices=tuple(variant.gripper_indices),
        arm_dof=variant.arm_dof,
        control_hz=variant.control_hz,
        is_dual_arm=False,  # 单臂配置（与 inference.py 对齐）
    )
    if len(robot_config["image_order"]) != variant.num_cameras:
        raise ValueError(
            "num_cameras must match the length of image_order: "
            f"got {variant.num_cameras} vs {len(robot_config['image_order'])}"
        )

    # ========== 3. 初始化 RL Agent（PixelSAC：图像编码器 + SAC） ==========
    # DummyEnv 提供占位的 observation/action space，用于初始化神经网络形状
    # 实际观测包含：
    #   - pixels: [H, W, C*num_cameras, 1] 拼接的多摄像头图像
    #   - state:  [proprio_dim + img_feature_dim, 1] 关节+夹爪+视觉嵌入
    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    logging.info("sample obs shapes %s", [(k, v.shape) for k, v in sample_obs.items()])
    logging.info("sample action shape %s", sample_action.shape)

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)  # 初始化 actor/critic 网络

    if variant.restore_path != "":
        logging.info("restoring from %s", variant.restore_path)
        agent.restore_checkpoint(variant.restore_path)

    online_buffer_size = 2 * variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space,
        dummy_env.action_space,
        int(online_buffer_size),
    )
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(
        variant,
        agent,
        env,
        eval_env,
        online_replay_buffer,
        replay_buffer,
        wandb_logger,
        shard_fn=shard_fn,
        agent_dp=agent_dp,
        robot_config=robot_config,
    )
