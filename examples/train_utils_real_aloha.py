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

try:
    from openpi_client import image_tools
except ImportError:
    from openpi.client import image_tools


def trajwise_alternating_training_loop(
    variant,
    agent,          # PixelSACLearner：在线 RL agent
    env,            # AgileXFollowerEnv 或 AlohaRobotEnv
    eval_env,
    online_replay_buffer,  # 存储在线采集的 transitions
    replay_buffer,
    wandb_logger,
    *,
    shard_fn=None,  # 多设备数据分片函数
    agent_dp=None,  # LocalPolicyClient：pi05 策略推理接口
    robot_config=None,  # 摄像头别名、控制频率等配置字典
):
    """轨迹级交替训练主循环。
    
    流程：
      1. 采集一条完整轨迹（人工标注成功/失败）
      2. 存入 replay buffer
      3. 执行多步 SAC 梯度更新（multi_grad_step * traj_length）
      4. 循环直到达到 max_steps
    
    与纯离线训练的区别：
      - 每条轨迹后立即更新策略，而非先收集大量数据
      - 需要平衡探索（SAC噪声）和利用（pi0策略引导）
    """
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)  # 分片到多 GPU

    i = 0  # 总训练步数
    total_env_steps = 0  # 累计环境交互步数
    total_num_traj = 0   # 累计轨迹数
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
    """将采集的轨迹转换为 transition 并存入 replay buffer。
    
    关键处理：
      - discount_horizon = query_freq（多步折扣）
      - 每个 (obs, action, reward, next_obs) 作为一条 transition
      - SAC 训练时使用这些 transitions 更新价值函数和策略
    """
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
    agent,      # PixelSACLearner：提供探索噪声
    env,        # 硬件环境
    i,          # 当前训练步数
    agent_dp=None,     # LocalPolicyClient：pi0 策略
    wandb_logger=None,
    traj_id=None,      # 轨迹 ID（用于视频保存）
    robot_config=None,
):
    """采集一条完整轨迹（从 reset 到人工标注结束）。
    
    采集策略：
      - 每 query_freq 步重新查询 pi0 策略（得到 chunk_size 个动作）
      - 对 pi0 输出叠加 SAC 探索噪声（agent.sample_actions）
      - 执行动作并记录观测、奖励（基于最终成功标签）
    
    返回：
      {
        "observations": List[obs_dict],  # 每步观测（pixels + state）
        "actions": List[action],         # RL agent 动作（query_freq 级别）
        "rewards": np.ndarray,           # 奖励信号（稀疏：成功 0，失败 -1）
        "masks": np.ndarray,             # episode 终止标记
        "is_success": bool,              # 人工标注的成功标志
        "env_steps": int                 # 低级控制步数
      }
    """
    query_frequency = variant.query_freq  # pi0 策略查询频率（如 25）
    instruction = variant.instruction     # 任务提示词
    max_timesteps = robot_config["max_timesteps"]  # 单轨迹最大步数
    agent._rng, rng = jax.random.split(agent._rng)  # JAX 随机数生成器

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
                    # 初始阶段：生成随机噪声 (1, 1, action_dim)
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    # repeat 到 action_horizon（pi05_agileX 为 50）
                    action_horizon = getattr(variant, "action_horizon", 50)
                    noise_repeat = jax.numpy.repeat(noise[:, -1:, :], action_horizon - noise.shape[1], axis=1)
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                    actions_noise = noise[0, : agent.action_chunk_shape[0], :]
                else:
                    # 后续阶段：使用 SAC 策略生成噪声
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    # repeat 到 action_horizon（pi05_agileX 为 50）
                    action_horizon = getattr(variant, "action_horizon", 50)
                    noise = np.repeat(actions_noise[-1:, :], action_horizon - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]

            # action_t = action[t % query_frequency]

            # # 对于 AgileX 机械臂，策略输出通常是原始的电机脉冲数值（raw counts），
            # # 因此不能进行 [-1, 1] 的截断，也不能简单地二值化夹爪（除非策略输出就是二值的）。
            # # 我们直接使用策略输出的动作。
            # # for idx in robot_config["gripper_indices"]:
            # #     if action_t[idx].item() > 0.5:
            # #         action_t[idx] = 1.0
            # #     else:
            # #         action_t[idx] = 0.0

            # # action_t = np.clip(action_t, -1, 1)
            
            action_t = action[t % query_frequency]

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
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == "c":
                    print("Continuing to next episode...")
                    break
            time.sleep(0.01)

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
    """从环境观测中提取并标准化图像、关节状态。
    
    处理步骤：
      1. 图像：直接使用 image_order 中的相机名称（camera0, camera1, ...）
      2. 关节状态：单臂 7-DOF（6 关节 + 1 夹爪）或双臂 14-DOF（取决于 is_dual_arm）
      3. 返回统一格式的观测字典
    """
    image_observations = obs_dict["images"]
    processed = {}
    
    # 直接使用 image_order 中的相机名称（与 inference.py 一致）
    for cam_name in robot_config["image_order"]:
        img = image_observations.get(cam_name)
        if img is None:
            raise ValueError(f"Camera {cam_name} not found in observation")
        # 转换通道顺序（如果是 CHW 格式）
        if img.shape[0] in (3, 4):
            img = img.transpose(1, 2, 0)
        # 只保留 RGB 通道
        img = img[..., :3]
        # 注意：Orbbec 相机通过 openpi 的 third_party 返回的是 RGB 格式
        # 不需要 BGR -> RGB 转换（与 inference.py 保持一致）
        # 如果使用 OpenCV 相机，则需要取消下行注释：
        # img = img[..., ::-1]  # BGR -> RGB (仅 OpenCV)
        processed[f"{cam_name}_image"] = img

    qpos = np.asarray(obs_dict["state"], dtype=np.float32)
    arm_dof = robot_config.get("arm_dof", 6)
    is_dual_arm = robot_config.get("is_dual_arm", True)  # 默认双臂（向后兼容）
    
    if is_dual_arm:
        # 双臂配置：14-DOF = 左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪
        left_joints = qpos[:arm_dof]
        left_gripper = qpos[arm_dof]
        right_joints = qpos[arm_dof + 1 : arm_dof * 2 + 1]
        right_gripper = qpos[-1]
        joint_position = np.concatenate([left_joints, right_joints])
        gripper_position = np.array([left_gripper, right_gripper])
    else:
        # 单臂配置：7-DOF = 6 关节 + 1 夹爪（与 inference.py 对齐）
        joint_position = qpos[:arm_dof]
        gripper_position = np.array([qpos[arm_dof]])

    processed["cartesian_position"] = np.zeros(robot_config.get("cartesian_dim", 6))
    processed["joint_position"] = joint_position
    processed["gripper_position"] = gripper_position
    return processed


def get_pi0_input(obs, robot_config, instruction):
    """构造 pi05 策略的输入字典（与 inference.py 格式完全一致）。
    
    注意：
      - inference.py 中直接使用 robot.get_observation() 返回的原始观测
      - 策略内部的 AgileXInputs transform 期望格式：
        {
            "state": np.array([7,]),
            "images": {"camera0": img_CHW, "camera1": img_CHW, ...},
            "prompt": "task description",  # 可选，如果没有会使用 default_prompt
        }
      - 图像格式：CHW (通道在前) 或 HWC (通道在后)，transform 内部会自动转换
    """
    # 构造与 inference.py 相同格式的观测字典
    # 注意：需要将 HWC 图像转换为 CHW 格式以匹配 AlohaAgileXFollower 的输出
    images = {}
    for cam_name in robot_config["image_order"]:
        img_key = f"{cam_name}_image"
        if img_key in obs:
            # obs 中的图像已经是 HWC 格式（经过 _extract_observation 处理）
            # resize 并保持 HWC 格式（AgileXInputs 内部会处理格式转换）
            img = image_tools.resize_with_pad(obs[img_key], 224, 224)
            images[cam_name] = img
    
    # 拼接 state（与 inference.py 一致）
    state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
    
    request_data = {
        "state": state,
        "images": images,
        "prompt": instruction,
    }
    
    return request_data


def process_images(variant, obs, robot_config):
    """将多个相机的图像按 image_order 拼接为 SAC agent 输入。

    步骤：
        1. 按 image_order 顺序（camera0, camera1, camera2, camera3）逐个 resize+pad 为固定分辨率
        2. 在通道维度上拼接（N 个相机 -> 3*N 通道）
        3. 扩展 batch 维度和 chunk 维度，得到 [1, H, W, 3N, 1]
    """
    resized = []
    for cam_name in robot_config["image_order"]:
        resized.append(image_tools.resize_with_pad(
            obs[f"{cam_name}_image"], variant.resize_image, variant.resize_image
        ))
    img_all = np.concatenate(resized, axis=2)[np.newaxis, ..., np.newaxis]
    return img_all
