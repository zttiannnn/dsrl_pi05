from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
import jax
import numpy as np
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

from openpi_client import image_tools


def trajwise_alternating_training_loop(
    variant,
    agent,
    env,
    eval_env,
    online_replay_buffer,
    replay_buffer,
    wandb_logger,
    *,
    shard_fn=None,
    agent_dp=None,
    robot_config=None,
):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    i = 0
    total_env_steps = 0
    total_num_traj = 0
    wandb_logger.log({"num_online_samples": 0}, step=i)
    wandb_logger.log({"num_online_trajs": 0}, step=i)
    wandb_logger.log({"env_steps": 0}, step=i)

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(
                variant,
                agent,
                env,
                i,
                agent_dp,
                wandb_logger,
                total_num_traj,
                robot_config,
            )
            total_num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj["env_steps"]
            print("online buffer timesteps length:", len(online_replay_buffer))
            print("online buffer num traj:", total_num_traj)
            print("total env steps:", total_env_steps)

            if i == 0:
                num_gradsteps = variant.multi_grad_step
            else:
                num_gradsteps = variant.multi_grad_step
            print(f"num_gradsteps: {num_gradsteps}")

            if total_num_traj >= variant.num_initial_traj_collect:
                for grad_step in range(num_gradsteps):
                    batch = next(replay_buffer_iterator)
                    if shard_fn is not None:
                        agent.update(batch)
                    else:
                        agent.update(batch)
                    i += 1
                    pbar.update(1)

            wandb_logger.log(
                {
                    "num_online_samples": len(online_replay_buffer),
                    "num_online_trajs": total_num_traj,
                    "env_steps": total_env_steps,
                },
                step=i,
            )


def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    discount_horizon = variant.query_freq
    actions = np.array(traj["actions"])
    episode_len = len(actions)
    rewards = np.array(traj["rewards"])
    masks = np.array(traj["masks"])

    for t in range(episode_len):
        obs = traj["observations"][t]
        next_obs = traj["observations"][t + 1]
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop("state", None)
            next_obs.pop("state", None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon,
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()


def collect_traj(
    variant,
    agent,
    env,
    i,
    agent_dp=None,
    wandb_logger=None,
    traj_id=None,
    robot_config=None,
):
    query_frequency = variant.query_freq
    instruction = variant.instruction
    max_timesteps = robot_config["max_timesteps"]
    agent._rng, rng = jax.random.split(agent._rng)

    try:
        env.reset()
    except Exception as exc:  # pragma: no cover - hardware failure path
        print("Environment reset failed")
        import traceback

        traceback.print_exc()
        raise exc

    step_time = 1 / robot_config.get("control_hz", 15)
    last_step_time = time.time()
    old_settings = termios.tcgetattr(sys.stdin)

    rewards = []
    action_list = []
    obs_list = []
    image_list = []
    is_success = False

    try:
        tty.setcbreak(sys.stdin.fileno())
        for t in tqdm(range(max_timesteps)):
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == "q":
                    print("'q' pressed, stopping loop.")
                    break

            try:
                _env_obs = env.get_observation()
            except Exception as exc:  # pragma: no cover
                print("Environment get obs failed")
                import traceback

                traceback.print_exc()
                raise exc

            curr_obs = _extract_observation(robot_config, _env_obs)
            image_list.append(curr_obs[robot_config["external_camera"] + "_image"])

            request_data = get_pi0_input(curr_obs, robot_config, instruction)

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)
                img_all = process_images(variant, curr_obs, robot_config)
                img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
                img_rep_pi0 = img_rep_pi0[:, -1, :]
                qpos = np.concatenate(
                    [curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()]
                )

                obs_dict = {
                    "pixels": img_all,
                    "state": qpos[np.newaxis, ..., np.newaxis],
                }

                if i == 0:
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 10 - noise.shape[1], axis=1)
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                    actions_noise = noise[0, : agent.action_chunk_shape[0], :]
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], 10 - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]

            action_t = action[t % query_frequency]

            for idx in robot_config["gripper_indices"]:
                if action_t[idx].item() > 0.5:
                    action_t[idx] = 1.0
                else:
                    action_t[idx] = 0.0

            action_t = np.clip(action_t, -1, 1)

            try:
                env.step(action_t)
            except Exception as exc:  # pragma: no cover
                print("Environment step failed")
                import traceback

                traceback.print_exc()
                raise exc

            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now

        print("Trial finished. Mark as (1) Success or (0) Failure:")
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input == "1":
                    print("Trial marked as SUCCESS.")
                    is_success = True
                    break
                if char_input == "0":
                    print("Trial marked as FAILURE.")
                    is_success = False
                    break
                print("Invalid input. Please enter '1' or '0':")
            time.sleep(0.01)

        try:
            _env_obs = env.get_observation()
        except Exception as exc:  # pragma: no cover
            print("Environment get obs failed")
            import traceback

            traceback.print_exc()
            raise exc

        curr_obs = _extract_observation(robot_config, _env_obs)
        image_list.append(curr_obs[robot_config["external_camera"] + "_image"])
        request_data = get_pi0_input(curr_obs, robot_config, instruction)
        img_all = process_images(variant, curr_obs, robot_config)
        img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
        img_rep_pi0 = img_rep_pi0[:, -1, :]
        qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])
        obs_dict = {
            "pixels": img_all,
            "state": qpos[np.newaxis, ..., np.newaxis],
        }
        obs_list.append(obs_dict)
        print("Rollout Done")

    finally:
        query_steps = len(action_list)
        if query_steps == 0:
            rewards = np.array([])
            masks = np.array([])
        elif is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

        if wandb_logger is not None:
            wandb_logger.log({"is_success": int(is_success)}, step=i)
            wandb_logger.log({"total_num_traj": traj_id}, step=i)

        video_path = os.path.join(variant.outputdir, f"video_high_{traj_id}.mp4")
        video = np.stack(image_list)
        ImageSequenceClip(list(video), fps=15).write_videofile(video_path, codec="libx264")

        print("Episode Done! Press c after resetting the environment")
        try:
            env.reset()
        except Exception as exc:  # pragma: no cover
            print("Environment reset failed")
            import traceback

            traceback.print_exc()
            raise exc
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    traj = {
        "observations": obs_list,
        "actions": action_list,
        "rewards": rewards,
        "masks": masks,
        "is_success": is_success,
        "env_steps": t + 1,
    }

    return traj


def _extract_observation(robot_config, obs_dict):
    image_observations = obs_dict["images"]
    processed = {}
    for alias, cam_name in robot_config["camera_map"].items():
        img = image_observations.get(cam_name)
        if img is None:
            raise KeyError(f"Camera {cam_name} missing from observation")
        if img.shape[0] in (3, 4):
            img = np.transpose(img, (1, 2, 0))
        img = img[..., :3]
        img = img[..., ::-1]
        processed[f"{alias}_image"] = img

    qpos = np.asarray(obs_dict["state"], dtype=np.float32)
    arm_dof = robot_config.get("arm_dof", 6)
    left_joints = qpos[:arm_dof]
    left_gripper = qpos[arm_dof]
    right_joints = qpos[arm_dof + 1 : arm_dof * 2 + 1]
    right_gripper = qpos[-1]
    joint_position = np.concatenate([left_joints, right_joints])
    gripper_position = np.array([left_gripper, right_gripper])

    processed["cartesian_position"] = np.zeros(robot_config.get("cartesian_dim", 6))
    processed["joint_position"] = joint_position
    processed["gripper_position"] = gripper_position
    return processed


def get_pi0_input(obs, robot_config, instruction):
    external_key = robot_config["external_camera"]
    wrist_key = robot_config["wrist_camera"]
    request_data = {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            obs[f"{external_key}_image"], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            obs[f"{wrist_key}_image"], 224, 224
        ),
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": instruction,
    }
    return request_data


def process_images(variant, obs, robot_config):
    resized = []
    for key in robot_config["image_order"]:
        resized.append(image_tools.resize_with_pad(obs[f"{key}_image"], variant.resize_image, variant.resize_image))
    img_all = np.concatenate(resized, axis=2)[np.newaxis, ..., np.newaxis]
    return img_all
