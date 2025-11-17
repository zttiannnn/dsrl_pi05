#! /usr/bin/env python
import os
import sys
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

OPENPI05_SRC = os.path.join(os.path.dirname(__file__), '..', 'openpi05', 'src')
if os.path.isdir(OPENPI05_SRC) and OPENPI05_SRC not in sys.path:
    sys.path.append(OPENPI05_SRC)

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

try:
    from openpi05.training import config as openpi05_config
    from openpi05.policies import policy_config as openpi05_policy_config
    from openpi05.shared import download as openpi05_download
except ImportError:
    openpi05_config = None
    openpi05_policy_config = None
    openpi05_download = None

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env in ('aloha_cube', 'aloha_cube_pi05'):
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32) # 32 is the noise action space of pi 0


def main(variant):
    variant.visualize_env = bool(getattr(variant, 'visualize_env', 0))
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    # outputdir = os.path.join(os.environ['EXP'], expname)
    outputdir = os.path.abspath(os.path.join(os.environ['EXP'], expname))
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    policy_module = policy_config
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]()
        task_id = 57
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
        if variant.visualize_env:
            print('LIBERO env does not expose a MuJoCo viewer; disabling visualization flag.')
            variant.visualize_env = False
    elif variant.env in ('aloha_cube', 'aloha_cube_pi05'):
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        render_mode = "human" if variant.visualize_env else "rgb_array"
        env = gym.make(
            "gym_aloha/AlohaTransferCube-v0",
            obs_type="pixels_agent_pos",
            render_mode=render_mode,
        )
        eval_env = gym.make(
            "gym_aloha/AlohaTransferCube-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
        if variant.visualize_env:
            print('MuJoCo viewer enabled; make sure MUJOCO_GL=glfw and DISPLAY is set.')
        variant.env_max_reward = 4
        variant.max_timesteps = 400
    else:
        if variant.visualize_env:
            print('Visualization is only implemented for Aloha environments; disabling flag.')
            variant.visualize_env = False
        

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    

    if variant.env == 'libero':
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    elif variant.env == 'aloha_cube_pi05':
        if openpi05_config is None or openpi05_policy_config is None or openpi05_download is None:
            raise ImportError("openpi05 未正确安装或未添加到 PYTHONPATH。请确认 openpi05/src 可被导入。")
        config = openpi05_config.get_config("pi05_aloha_sim")
        checkpoint_dir = openpi05_download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
        policy_module = openpi05_policy_config
    else:
        raise NotImplementedError()
    agent_dp = policy_module.create_trained_policy(config, checkpoint_dir)
    print("Loaded policy from %s", checkpoint_dir)
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
 