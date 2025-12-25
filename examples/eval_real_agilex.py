#! /usr/bin/env python
"""Entry point for running pi05 / Aloha real-world evaluation."""

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
import collections
import multiprocessing as mp
from functools import partial

import gymnasium as gym
import jax
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Dict
from jax.experimental.compilation_cache import compilation_cache
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

from jaxrl2.utils.launch_util import parse_training_args
from jaxrl2.utils.wandb_logger import create_exp_name

from examples.agilex_env_wrapper import AgileXFollowerEnv
from examples.train_utils_real_aloha import get_pi0_input, process_images, _extract_observation
from examples.train_real_aloha import DummyEnv, shard_batch

def _apply_transition(old_actions, new_actions, h_fn):
    """
    通用 transition 应用器：对前 n_interp 个旧动作与新动作按权重 h(t) 做插值。
    h_fn: 接受 t in (0,1) 返回权重 h(t) 的函数，interp = (1-h)*old + h*new。
    返回 list[np.ndarray]
    """
    n_old = len(old_actions)
    n_interp = min(n_old, len(new_actions))
    result = []
    for i_interp in range(n_interp):
        t = (i_interp + 1) / (n_interp + 1)
        h = float(h_fn(t))
        interp_action = (1 - h) * old_actions[i_interp] + h * new_actions[i_interp]
        result.append(interp_action)
    for a in new_actions[n_interp:]:
        result.append(a)
    return result

def cubic_transition(old_actions, new_actions):
    """三次 Hermite ease-in/out：h(t)=3t^2-2t^3"""
    return _apply_transition(old_actions, new_actions, lambda t: 3 * t**2 - 2 * t**3)


def smooth_horizon(actions: np.ndarray, method: str = "none", window: int = 3, ema_alpha: float = 0.9):
    """
    Smooth predicted horizon (actions: shape (H, D)). Returns smoothed array same shape.
    method: 'none'|'moving'|'median'|'ema'
    window: integer window size for moving/median (should be odd for median/centered moving)
    ema_alpha: smoothing factor for EMA (0-1)
    """
    if method == "none" or window <= 1 and method in ("moving", "median"):
        return actions
    actions = np.asarray(actions, dtype=float)
    H, D = actions.shape
    if method == "moving":
        k = max(1, int(window))
        if k == 1:
            return actions
        kernel = np.ones(k, dtype=float) / k
        sm = np.zeros_like(actions)
        for d in range(D):
            sm[:, d] = np.convolve(actions[:, d], kernel, mode="same")
        return sm
    elif method == "median":
        k = max(1, int(window))
        if k == 1:
            return actions
        pad = k // 2
        padded = np.pad(actions, ((pad, pad), (0, 0)), mode="edge")
        sm = np.zeros_like(actions)
        for t in range(H):
            sm[t] = np.median(padded[t:t + k], axis=0)
        return sm
    elif method == "ema":
        alpha = float(ema_alpha)
        sm = np.zeros_like(actions)
        s = actions[0].copy()
        for t in range(H):
            s = alpha * actions[t] + (1.0 - alpha) * s
            sm[t] = s
        return sm
    else:
        raise ValueError(f"Unknown horizon smoothing method: {method!r}")


def _build_D2(H: int) -> np.ndarray:
    """Construct second-difference matrix D2 of shape (H-2, H).
    (D2 a)[t] = a[t+2] - 2 a[t+1] + a[t]
    """
    if H < 3:
        return np.zeros((0, H), dtype=float)
    D2 = np.zeros((H - 2, H), dtype=float)
    for i in range(H - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


def optimize_horizon_qp(new_actions: np.ndarray, lambda_acc: float = 0.1, velocity_limit: float = 0.0, anchor: np.ndarray | None = None) -> np.ndarray:
    """
    Light-weight QP-style optimizer: minimize 0.5||a - a_ref||^2 + 0.5 * lambda_acc * ||D2 a||^2
    Solves per-joint linear system: (I + lambda_acc * D2^T D2) a = a_ref
    new_actions: (H, D)
    velocity_limit: if >0, post-clamp per-step deltas to [-velocity_limit, velocity_limit]
    anchor: optional previous executed action (1D array of length D) used to bias first element
    Returns optimized actions (H, D)
    """
    a_ref = np.asarray(new_actions, dtype=float)
    H, D = a_ref.shape
    if H == 0:
        return a_ref
    D2 = _build_D2(H)
    if lambda_acc <= 0 or D2.size == 0:
        sol = a_ref.copy()
    else:
        M = np.eye(H, dtype=float)
        # add regularizer: lambda_acc * D2^T D2
        reg = lambda_acc * (D2.T @ D2)
        A = M + reg
        # For numerical stability, add small diag jitter
        A += np.eye(H) * 1e-8
        sol = np.zeros_like(a_ref)
        # Solve per joint
        for j in range(D):
            b = a_ref[:, j]
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            sol[:, j] = x
    # Optional simple post-processing: clamp per-step velocity
    if velocity_limit and velocity_limit > 0.0:
        # if anchor provided, use it as previous value; else use sol[0]
        prev = None
        if anchor is not None:
            prev = np.asarray(anchor, dtype=float)
        else:
            prev = sol[0].copy()
        # enforce on each joint independently
        for j in range(D):
            val = prev[j]
            for t in range(H):
                delta = sol[t, j] - val
                if delta > velocity_limit:
                    delta = velocity_limit
                elif delta < -velocity_limit:
                    delta = -velocity_limit
                val = val + delta
                sol[t, j] = val
    return sol


# ---------- 子进程：推理循环（包含 RL agent 和 pi05） ----------
def inference_worker(
    in_q: mp.Queue,
    out_q: mp.Queue,
    checkpoint: str,
    config_name: str,
    default_prompt: str,
    rl_restore_path: str,
    rl_kwargs: dict,
    rl_seed: int,
    sample_obs_shape: dict,
    sample_action_shape: tuple,
    action_horizon: int,
    guidance_sigma: float = 0.2,
):
    """
    推理子进程：加载 pi05 policy 和 RL agent，持续从 in_q 获取观测数据并推理。
    所有推理都在子进程中完成，主进程完全不阻塞。
    
    RTC (Real-Time Action Chunking) 支持:
    - 接收 constraint_actions (前一个 chunk 的未执行尾部) 用于 inpainting
    - 使用 guidance_sigma (Smooth-as-Butter 参数) 进行更紧密的引导
    """
    # 在子进程中导入和初始化，避免 CUDA context 冲突
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    
    from examples.local_policy_client import LocalPolicyClient
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
    from jaxrl2.utils.general_utils import add_batch_dim
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Inference worker: initializing...")
    
    # 1. 初始化 pi05 policy
    agent_dp = LocalPolicyClient(checkpoint, config_name=config_name, default_prompt=default_prompt)
    logging.info("Inference worker: LocalPolicyClient initialized")
    
    # 2. 初始化 RL agent
    # 重建 sample_obs 和 sample_action
    sample_obs = {}
    for k, v in sample_obs_shape.items():
        sample_obs[k] = np.zeros(v["shape"], dtype=np.dtype(v["dtype"]))
    sample_obs = add_batch_dim(sample_obs)
    sample_action = np.zeros(sample_action_shape, dtype=np.float32)
    sample_action = add_batch_dim(sample_action)
    
    rl_agent = PixelSACLearner(rl_seed, sample_obs, sample_action, **rl_kwargs)
    if rl_restore_path:
        logging.info("Inference worker: restoring RL agent from %s", rl_restore_path)
        rl_agent.restore_checkpoint(rl_restore_path)
    else:
        logging.warning("Inference worker: no RL restore_path, using random weights")
    logging.info("Inference worker: RL agent initialized")
    
    while True:
        item = in_q.get()
        if item is None:  # 收到结束标识
            del agent_dp
            del rl_agent
            break
        
        # item = (idx, request_data, img_all, qpos_base, constraint_actions)
        # qpos_base = (joint_position, gripper_position) 不含 img_rep
        # constraint_actions = 前一个 chunk 的未执行尾部 (用于 RTC inpainting)
        idx, request_data, img_all, qpos_base, constraint_actions = item
        
        total_start = time.time()
        
        # 1. 获取 pi05 prefix representation
        t0 = time.time()
        img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
        img_rep_pi0 = img_rep_pi0[:, -1, :]
        prefix_time = time.time() - t0
        
        # 2. 构建完整的 qpos（包含 img_rep）
        qpos = np.concatenate([qpos_base, img_rep_pi0.flatten()])
        
        obs_dict = {
            "pixels": img_all,
            "state": qpos[np.newaxis, ..., np.newaxis],
        }
        
        # 3. RL agent 推理
        t0 = time.time()
        if hasattr(rl_agent, "eval_actions"):
            actions_residual = rl_agent.eval_actions(obs_dict)
        else:
            actions_residual = rl_agent.sample_actions(obs_dict)
        rl_time = time.time() - t0
        
        actions_residual = np.reshape(actions_residual, rl_agent.action_chunk_shape)
        
        # Repeat to match action horizon
        noise_repeat = np.repeat(actions_residual[-1:, :], action_horizon - actions_residual.shape[0], axis=0)
        noise = jnp.concatenate([actions_residual, noise_repeat], axis=0)[None]
        
        # 4. pi05 推理 (with RTC constraints)
        t0 = time.time()
        result = agent_dp.infer(
            request_data,
            noise=np.asarray(noise),
            constraint_actions=constraint_actions,  # RTC: inpainting constraint
            guidance_sigma=guidance_sigma,  # Smooth-as-Butter: tighter guidance
        )
        pi05_time = time.time() - t0
        
        total_time = time.time() - total_start
        constraint_len = 0 if constraint_actions is None else len(constraint_actions)
        logging.info(f"Inference worker step {idx}: prefix={prefix_time:.3f}s, RL={rl_time:.3f}s, pi05={pi05_time:.3f}s, total={total_time:.3f}s, constraint_len={constraint_len}")
        
        out_q.put((idx, result["actions"], actions_residual))


home_dir = os.environ["HOME"]
compilation_cache.initialize_cache(os.path.join(home_dir, "jax_compilation_cache"))


def eval_policy(variant, env, in_q, out_q, robot_config, sent_idx_ref):
    action_steps = getattr(variant, "action_steps", 15)
    instruction = variant.instruction
    max_timesteps = robot_config["max_timesteps"]
    use_rtc = getattr(variant, "use_rtc", True)  # RTC enabled by default
    
    print("Resetting environment...")
    try:
        env.reset()
    except Exception as exc:
        print("Environment reset failed")
        import traceback
        traceback.print_exc()
        raise exc

    step_time = 1 / robot_config.get("control_hz", 15)
    old_settings = termios.tcgetattr(sys.stdin)

    action_list = []
    image_list = []
    action_queue = collections.deque()
    waiting_for_infer = False
    action_step_counter = 0
    first_inference = True
    last_action_t = None  # 保存上一个执行的动作
    
    # RTC (Real-Time Action Chunking) state
    last_full_chunk = None  # 完整的上一个 action chunk (用于计算 constraint)
    last_request_step = 0   # 上一次发送推理请求时的步数 (用于计算 steps_consumed)
    current_step = 0        # 当前执行的总步数
    
    print("Starting evaluation loop. Press 'q' to stop.")
    print(f"RTC mode: {'enabled' if use_rtc else 'disabled'}")
    try:
        tty.setcbreak(sys.stdin.fileno())
        for t in tqdm(range(max_timesteps)):
            loop_start = time.perf_counter()
            
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == "q":
                    print("'q' pressed, stopping loop.")
                    break

            try:
                _env_obs = env.get_observation()
            except Exception as exc:
                print("Environment get obs failed")
                import traceback
                traceback.print_exc()
                raise exc

            curr_obs = _extract_observation(robot_config, _env_obs)
            image_list.append(curr_obs[robot_config["external_camera"] + "_image"])

            # 1. 尝试获取推理结果（非阻塞）- 移到发起请求之前
            try:
                idx, new_actions, actions_residual = out_q.get_nowait()
                logging.debug(f"got result #{idx}")
                
                # 保存 RL 动作用于后续分析
                action_list.append(actions_residual)
                
                # RTC: 保存完整的新 chunk 用于下一次计算 constraint
                new_actions = np.asarray(new_actions, dtype=float)
                if new_actions.ndim == 1:
                    new_actions = new_actions[None, :]
                last_full_chunk = new_actions.copy()
                
                # 获取旧动作用于可选的后处理
                old_actions = list(action_queue)
                action_queue.clear()
                
                # Horizon-level 平滑（对整个预测序列进行时序平滑）
                horizon_smooth = getattr(variant, "horizon_smooth", "none")
                horizon_window = getattr(variant, "horizon_window", 3)
                horizon_ema_alpha = getattr(variant, "horizon_ema_alpha", 0.9)
                
                if horizon_smooth != "none" and len(new_actions) > 0:
                    try:
                        arr = new_actions
                        H, D = arr.shape
                        # 分离 body (关节) 和 gripper
                        if D >= 2:
                            body = arr[:, :-1]
                            grip = arr[:, -1:]
                        else:
                            body = arr
                            grip = None
                        if body.size > 0:
                            body = smooth_horizon(body, method=horizon_smooth, window=horizon_window, ema_alpha=horizon_ema_alpha)
                        new_actions = np.concatenate([body, grip], axis=1) if grip is not None else body
                    except Exception as e:
                        logging.warning(f"Horizon smoothing failed: {e}")
                
                # QP 优化（最小化加速度/二阶差分）- 可选的安全过滤器
                qp_lambda_acc = getattr(variant, "qp_lambda_acc", 0.0)
                qp_velocity_limit = getattr(variant, "qp_velocity_limit", 0.0)
                
                if qp_lambda_acc > 0 and len(new_actions) > 0:
                    try:
                        # anchor 使用旧动作队列的第一个动作
                        anchor = None
                        if len(old_actions) > 0:
                            anchor = np.asarray(old_actions[0], dtype=float)
                        
                        arr = np.asarray(new_actions, dtype=float)
                        H, D = arr.shape
                        # 分离 body (关节) 和 gripper
                        if D >= 2:
                            body = arr[:, :-1]
                            grip = arr[:, -1:]
                        else:
                            body = arr
                            grip = None
                        if body.size > 0:
                            body = optimize_horizon_qp(body, lambda_acc=qp_lambda_acc, velocity_limit=qp_velocity_limit, anchor=anchor)
                        new_actions = np.concatenate([body, grip], axis=1) if grip is not None else body
                    except Exception as e:
                        logging.warning(f"QP optimization failed: {e}")
                
                # RTC: 不再需要 cubic_transition，因为 RTC inpainting 确保新 chunk 与旧 chunk 尾部对齐
                if use_rtc:
                    # RTC 模式：直接追加新动作（新 chunk 开头已与旧 chunk 尾部对齐）
                    action_queue.extend(new_actions)
                else:
                    # 非 RTC 模式：使用 cubic_transition 进行后处理平滑
                    if len(old_actions) == 0:
                        action_queue.extend(new_actions)
                    else:
                        smoothed_actions = cubic_transition(old_actions, new_actions)
                        action_queue.extend(smoothed_actions)
                
                waiting_for_infer = False
            except:
                pass  # 队列为空，继续

            # 2. 当执行了 action_steps 步后（或首次），发起新推理请求
            if not waiting_for_infer and (action_step_counter >= action_steps or first_inference):
                first_inference = False
                
                # 准备数据发送到子进程（不做任何推理，只做数据准备）
                request_data = get_pi0_input(curr_obs, robot_config, instruction)
                img_all = process_images(variant, curr_obs, robot_config)
                
                # qpos_base 不含 img_rep，img_rep 在子进程中计算
                qpos_base = np.concatenate(
                    [curr_obs["joint_position"], curr_obs["gripper_position"]]
                )
                
                # RTC: 计算 constraint_actions (前一个 chunk 的未执行尾部)
                constraint_actions = None
                if use_rtc and last_full_chunk is not None:
                    # 计算自上次发送请求以来消耗的步数
                    steps_consumed = current_step - last_request_step
                    if steps_consumed < len(last_full_chunk):
                        # 提取未执行的尾部: A_{t-1}[d:]
                        constraint_actions = last_full_chunk[steps_consumed:]
                        logging.debug(f"RTC constraint: steps_consumed={steps_consumed}, constraint_len={len(constraint_actions)}")
                    else:
                        logging.debug(f"RTC: all actions consumed (steps_consumed={steps_consumed} >= chunk_len={len(last_full_chunk)})")
                
                # 记录本次请求的步数
                last_request_step = current_step
                
                # 将推理请求发送到子进程（非阻塞）
                try:
                    in_q.put_nowait((sent_idx_ref[0], request_data, img_all, qpos_base, constraint_actions))
                    sent_idx_ref[0] += 1
                    waiting_for_infer = True
                    action_step_counter = 0
                except:
                    logging.debug("inference queue full, dropping frame")

            # 3. 执行动作 - 关键：无论如何都要在这一帧发送动作
            if len(action_queue) > 0:
                action_t = action_queue.popleft()
                action_step_counter += 1
                current_step += 1  # RTC: 跟踪总执行步数
                last_action_t = action_t  # 保存当前动作
            else:
                # 队列为空时，复用上一个动作（保持机器人运动）
                if last_action_t is not None:
                    action_t = last_action_t
                else:
                    action_t = np.zeros(7)  # Default action dim
                # 不增加 action_step_counter，这样下一帧会尝试发起推理

            try:
                env.step(action_t)
            except Exception as exc:
                print("Environment step failed")
                import traceback
                traceback.print_exc()
                raise exc

            # 4. 精确控制帧率
            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = step_time - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("Evaluation finished.")

        if len(image_list) > 0:
            video_path = os.path.join(variant.outputdir, f"eval_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            print(f"Saving video to {video_path}...")
            try:
                video = np.stack(image_list)
                ImageSequenceClip(list(video), fps=robot_config.get("control_hz", 30)).write_videofile(video_path, codec="libx264")
            except Exception as e:
                print(f"Failed to save video: {e}")
        
        print("Robot will auto-reset to home position. Press 'c' to confirm...")
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == "c":
                    print("Auto-resetting robot to home position...")
                    break
            time.sleep(0.01)

        try:
            env.reset()
        except Exception as exc:
            print("Environment reset failed")


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    logging.info("num devices %s", num_devices)
    
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = variant["train_kwargs"]

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    if "EXP" in os.environ:
        outputdir = os.path.join(os.environ["EXP"], expname)
    else:
        outputdir = os.path.join(os.getcwd(), "logs", expname)

    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print("writing to output dir", outputdir)
    
    # 1. 获取配置参数
    checkpoint = getattr(variant, "policy_checkpoint", "")
    if not checkpoint:
        raise ValueError("--policy_checkpoint must point to a trained pi05_agileX checkpoint")
    cfg_name = getattr(variant, "policy_config", "pi05_agileX")
    default_prompt = variant.instruction
    rl_restore_path = getattr(variant, "restore_path", "")
    action_horizon = getattr(variant, "action_horizon", 50)
    guidance_sigma = getattr(variant, "guidance_sigma", 0.2)  # Smooth-as-Butter parameter
    
    logging.info("Policy checkpoint: %s, config: %s", checkpoint, cfg_name)
    logging.info("RTC enabled: %s, guidance_sigma: %s", getattr(variant, "use_rtc", True), guidance_sigma)
    
    # 2. 准备 RL agent 的 observation/action space 信息（用于在子进程中重建）
    dummy_env = DummyEnv(variant)
    sample_obs_raw = dummy_env.observation_space.sample()
    sample_action_raw = dummy_env.action_space.sample()
    
    # 序列化 observation space 的 shape 和 dtype 信息
    sample_obs_shape = {}
    for k, v in sample_obs_raw.items():
        sample_obs_shape[k] = {
            "shape": v.shape,
            "dtype": str(v.dtype),
        }
    sample_action_shape = sample_action_raw.shape

    # 3. 启动推理子进程（包含 RL agent 和 pi05）
    ctx = mp.get_context("spawn")
    in_q: mp.Queue = ctx.Queue(maxsize=4)
    out_q: mp.Queue = ctx.Queue(maxsize=4)
    
    proc = ctx.Process(
        target=inference_worker,
        args=(
            in_q, out_q,
            checkpoint, cfg_name, default_prompt,
            rl_restore_path, kwargs, variant.seed,
            sample_obs_shape, sample_action_shape,
            action_horizon,
            guidance_sigma,  # Smooth-as-Butter parameter
        )
    )
    proc.daemon = False
    proc.start()
    logging.info("Inference worker process started (contains both RL agent and pi05)")

    # 4. Initialize Environment
    robot_type = getattr(variant, "robot_type", variant.env).lower()
    logging.info("Selected robot type: %s", robot_type)
    
    if robot_type == "agilex":
        if not variant.agilex_port:
            raise ValueError("--agilex_port must be provided when robot_type=agilex")
        env = AgileXFollowerEnv(
            port=variant.agilex_port,
            robot_id=getattr(variant, "agilex_robot_id", "left"),
            camera_config=getattr(variant, "agilex_camera_dict", None),
            camera_config_path=None,
            max_relative_target=getattr(variant, "agilex_max_relative_target", None),
            use_degrees=bool(getattr(variant, "agilex_use_degrees", False)),
            prompt=variant.instruction,
        )
    else:
        raise ValueError(f"Unsupported robot type for evaluation: {robot_type}")

    robot_config = dict(
        image_order=list(variant.image_order),
        external_camera=variant.image_order[0],
        max_timesteps=variant.real_env_max_steps,
        gripper_indices=tuple(variant.gripper_indices),
        arm_dof=variant.arm_dof,
        control_hz=variant.control_hz,
        is_dual_arm=False,
    )

    # 5. Run Evaluation Loop（主进程不再有 RL agent）
    sent_idx_ref = [0]  # 使用列表来允许在函数中修改
    num_eval_episodes = getattr(variant, "eval_episodes", 10)
    for i in range(num_eval_episodes):
        print(f"Starting evaluation episode {i+1}/{num_eval_episodes}")
        eval_policy(variant, env, in_q, out_q, robot_config, sent_idx_ref)
        
        if i < num_eval_episodes - 1:
            print("Press 'n' for next episode, or 'q' to quit evaluation.")
            stop_eval = False
            while True:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    char_input = sys.stdin.read(1)
                    if char_input.lower() == "n":
                        break
                    if char_input.lower() == "q":
                        stop_eval = True
                        break
                time.sleep(0.01)
            if stop_eval:
                break

    # 6. 清理：通知子进程退出
    in_q.put(None)
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
    logging.info("Inference worker process terminated")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    
    # Add arguments similar to launch_train_real_aloha.py
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--env", default="agilex", help="Name of environment")
    parser.add_argument("--robot_type", default="agilex", choices=["agilex"], help="Robot type")
    parser.add_argument("--batch_size", default=256, type=int) # Needed for DummyEnv/Learner init
    parser.add_argument("--max_steps", default=int(5e5), type=int) # Needed for decay steps calculation
    parser.add_argument("--add_states", default=1, type=int)
    parser.add_argument("--resize_image", default=224, type=int)
    parser.add_argument("--query_freq", default=25, type=int)
    parser.add_argument("--task", default="pick up the circular chip and place it on the yellow pot")
    parser.add_argument("--policy_checkpoint", default="", help="Path to pi05 policy checkpoint")
    parser.add_argument("--policy_config", default="pi05_agileX")
    parser.add_argument("--proprio_dim", default=7, type=int)
    parser.add_argument("--img_feature_dim", default=2048, type=int)
    parser.add_argument("--arm_dof", default=6, type=int)
    parser.add_argument("--action_horizon", default=50, type=int)
    parser.add_argument("--image_order", nargs="+", default=["camera0", "camera1", "camera2", "camera3"])
    parser.add_argument("--gripper_indices", nargs="+", default=[-1], type=int)
    parser.add_argument("--real_env_max_steps", default=1000, type=int)
    parser.add_argument("--control_hz", default=30, type=int)
    parser.add_argument("--agilex_port", default="")
    parser.add_argument("--agilex_robot_id", default="left")
    parser.add_argument("--agilex_cameras_inline", default="")
    parser.add_argument("--agilex_max_relative_target", default=None, type=int)
    parser.add_argument("--agilex_use_degrees", action="store_true")
    parser.add_argument("--action_steps", default=20, type=int, help="Number of steps between inferences")
    
    # Horizon-level smoothing options (默认值与 inference_smooth_0912.py 一致)
    parser.add_argument("--horizon_smooth", type=str, default="ema", choices=["none", "moving", "median", "ema"], help="对预测 horizon 进行时序平滑: none/moving/median/ema")
    parser.add_argument("--horizon_window", type=int, default=30, help="窗口大小用于 moving/median 平滑（越大越平滑）")
    parser.add_argument("--horizon_ema_alpha", type=float, default=0.7, help="horizon EMA alpha 用于 horizon_smooth=ema")
    
    # QP-style optimizer options
    parser.add_argument("--qp_lambda_acc", type=float, default=0.0, help="二阶差分加速惩罚系数（>=0），qp优化时使用。0 表示禁用")
    parser.add_argument("--qp_velocity_limit", type=float, default=0.0, help="可选的每步最大速度（动作单位/step），>0 则启用速度束缚")
    
    # RTC (Real-Time Action Chunking) with Smooth-as-Butter options
    parser.add_argument("--use_rtc", action="store_true", default=True, help="启用 RTC (Real-Time Action Chunking) inpainting 模式")
    parser.add_argument("--no_rtc", action="store_false", dest="use_rtc", help="禁用 RTC，使用传统的 cubic_transition 后处理平滑")
    parser.add_argument("--guidance_sigma", type=float, default=0.2, help="Smooth-as-Butter 参数：较小的值 (如 0.2) 提供更紧密的引导约束")
    
    # Specific for evaluation
    parser.add_argument("--restore_path", default="", help="Path to trained RL agent checkpoint")
    parser.add_argument("--eval_episodes", default=10, type=int)

    # Dummy args for parse_training_args compatibility
    parser.add_argument("--launch_group_id", default="")
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--multi_grad_step", default=1, type=int)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--eval_interval", default=5000, type=int)
    parser.add_argument("--checkpoint_interval", default=-1, type=int)
    parser.add_argument("--num_initial_traj_collect", default=1, type=int)
    parser.add_argument("--algorithm", default="pixel_sac")

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(1024, 1024, 1024),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(3, 2, 2, 2),
        cnn_padding="VALID",
        latent_dim=50,
        discount=0.99,
        tau=0.005,
        critic_reduction="min",
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type="small",
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy=0.0,
        num_qs=2,
        action_magnitude=2.5,
        num_cameras=4,
    )

    # Need to import yaml for camera parsing
    import yaml
    
    variant, args = parse_training_args(train_args_dict, parser)

    # Parse camera config
    inline_spec = getattr(variant, "agilex_cameras_inline", "")
    variant.agilex_camera_dict = None
    if inline_spec:
        try:
            parsed = yaml.safe_load(inline_spec)
            if not isinstance(parsed, dict):
                raise ValueError("camera spec must decode to a mapping")
            variant.agilex_camera_dict = parsed
        except Exception as exc:
            raise SystemExit(f"Failed to parse --agilex_cameras_inline: {exc}")
            
    variant.instruction = getattr(variant, "task", "")
    
    main(variant)
