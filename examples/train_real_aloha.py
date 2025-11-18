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

from examples.aloha_env_wrapper import AlohaRobotEnv
from examples.train_utils_real_aloha import trajwise_alternating_training_loop
from openpi_client import websocket_client_policy as _websocket_client_policy

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

    # Policy connection: prefer a local inference process (inproc) if
    # environment variable `LOCAL_POLICY_CHECKPOINT` is set. This allows
    # running the model in a child process (same machine) instead of a
    # websocket server.
    agent_dp = None
    if os.environ.get("LOCAL_POLICY_CHECKPOINT"):
        from examples.local_policy_client import LocalPolicyClient

        checkpoint = os.environ["LOCAL_POLICY_CHECKPOINT"]
        cfg_name = os.environ.get("LOCAL_POLICY_CONFIG", "pi05_aloha")
        default_prompt = os.environ.get("LOCAL_POLICY_PROMPT", None)
        agent_dp = LocalPolicyClient(checkpoint, config_name=cfg_name, default_prompt=default_prompt)
        metadata = agent_dp.get_server_metadata()
        logging.info("Using LocalPolicyClient, metadata: %s", metadata)
    else:
        agent_dp = _websocket_client_policy.WebsocketClientPolicy(
            host=os.environ.get("remote_host", "0.0.0.0"),
            port=os.environ.get("remote_port", None),
        )
        metadata = agent_dp.get_server_metadata()
        logging.info("Using WebsocketClientPolicy, server metadata: %s", metadata)

    logging.info("initializing Aloha environment...")
    env = AlohaRobotEnv(render_size=variant.resize_image, reset_position=metadata.get("reset_pose"))
    eval_env = env
    logging.info("created the aloha env!")

    robot_config = dict(
        camera_map=dict(
            high="cam_high",#########################################3
            low="cam_low",
            left_wrist="cam_left_wrist",
            right_wrist="cam_right_wrist",
        ),########################################改成松灵的配置
        image_order=list(variant.image_order),
        external_camera=variant.external_camera,
        wrist_camera=variant.wrist_camera,
        max_timesteps=variant.real_env_max_steps,
        gripper_indices=tuple(variant.gripper_indices),
        arm_dof=variant.arm_dof,
        control_hz=variant.control_hz,
    )
    if len(robot_config["image_order"]) != variant.num_cameras:
        raise ValueError(
            "num_cameras must match the length of image_order: "
            f"got {variant.num_cameras} vs {len(robot_config['image_order'])}"
        )

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    logging.info("sample obs shapes %s", [(k, v.shape) for k, v in sample_obs.items()])
    logging.info("sample action shape %s", sample_action.shape)

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

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
