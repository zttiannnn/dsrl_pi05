import argparse
import sys
import os
from typing import Dict

# Ensure 'openpi' directory is in PYTHONPATH to allow importing 'third_party'
# Assuming this script is in examples/ and openpi/ is a sibling of examples/..
# i.e. dsrl_pi05/examples/launch_train_real_aloha.py -> dsrl_pi05/openpi
current_dir = os.path.dirname(os.path.abspath(__file__))
openpi_dir = os.path.join(os.path.dirname(current_dir), "openpi")
openpi_src_dir = os.path.join(openpi_dir, "src")
if openpi_dir not in sys.path:
    # Ensure src is preferred so we load the package from openpi/src/openpi
    if openpi_src_dir not in sys.path:
        sys.path.insert(0, openpi_src_dir)
    if openpi_dir not in sys.path:
        sys.path.insert(0, openpi_dir)

    # After import, make sure openpi.__path__ contains the third_party folder
    try:
        import importlib
        pkg = importlib.import_module("openpi")
        third_party = os.path.join(openpi_dir, "third_party")
        if os.path.isdir(third_party) and third_party not in getattr(pkg, "__path__", []):
            pkg.__path__.append(third_party)
    except Exception:
        # Best-effort only; failures will surface when importing specific modules
        pass
import yaml

from examples.train_real_aloha import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, help="Random seed.", type=int)
    parser.add_argument("--launch_group_id", default="", help="group id used to group runs on wandb.")
    parser.add_argument("--eval_episodes", default=10, help="Number of episodes used for evaluation.", type=int)
    parser.add_argument("--env", default="aloha", help="Name of environment")
    parser.add_argument(
        "--robot_type",
        default="aloha",
        choices=["aloha", "agilex"],
        help="Selects which hardware wrapper to launch.",
    )
    parser.add_argument("--log_interval", default=1000, help="Logging interval.", type=int)
    parser.add_argument("--eval_interval", default=5000, help="Eval interval.", type=int)
    parser.add_argument("--checkpoint_interval", default=-1, help="Checkpoint interval.", type=int)
    parser.add_argument("--batch_size", default=256, help="Mini batch size.", type=int)
    parser.add_argument("--max_steps", default=int(5e5), help="Number of training steps.", type=int)
    parser.add_argument(
        "--add_states",
        default=1,
        help="whether to add low-dim states to the observations",
        type=int,
    )
    parser.add_argument("--wandb_project", default="dsrl_pi05_real", help="wandb project")
    parser.add_argument(
        "--num_initial_traj_collect",
        default=1,
        help="number of trajectories to collect before starting online updates",
        type=int,
    )
    parser.add_argument("--algorithm", default="pixel_sac", help="type of algorithm")
    parser.add_argument("--prefix", default="", help="prefix to use for wandb")
    parser.add_argument("--suffix", default="", help="suffix to use for wandb")
    parser.add_argument(
        "--multi_grad_step",
        default=30,
        help="Number of gradient steps to take per environment step, aka UTD",
        type=int,
    )
    parser.add_argument(
        "--resize_image",
        default=224,
        help="the size of image if need resizing",
        type=int,
    )
    parser.add_argument("--query_freq", default=25, help="query frequency", type=int)
    parser.add_argument(
        "--task",
        default="pick up the circular chip and place it on the yellow pot",
        help="Language instruction / prompt shared with the policy (mirrors inference --task).",
    )
    parser.add_argument(
        "--policy_checkpoint",
        default="",
        help="Path to the pi05 policy checkpoint (same as inference --checkpoint_dir).",
    )
    parser.add_argument(
        "--policy_config",
        default="pi05_agileX",
        help="Policy config name passed to LocalPolicyClient (default: pi05_agileX).",
    )
    parser.add_argument("--proprio_dim", default=7, help="dimension of proprioceptive state (single arm: 6 joints + 1 gripper)", type=int)
    parser.add_argument("--img_feature_dim", default=2048, help="dimension of pi0 visual features", type=int)
    parser.add_argument("--arm_dof", default=6, help="Number of joints per arm", type=int)
    parser.add_argument("--action_horizon", default=50, help="pi05 action horizon (number of actions per chunk)", type=int)
    parser.add_argument(
        "--image_order",
        nargs="+",
        default=["camera0", "camera1", "camera2", "camera3"],
        help="ordering of cameras for pixel observations (matches inference.py camera naming)",
    )
    parser.add_argument(
        "--gripper_indices",
        nargs="+",
        default=[-1],
        type=int,
        help="indices inside the pi0 action vector corresponding to grippers (single arm: only last dim)",
    )
    parser.add_argument(
        "--real_env_max_steps",
        default=1000,
        type=int,
        help="maximum number of low-level control steps per real rollout",
    )
    parser.add_argument("--control_hz", default=30, type=int, help="command frequency for the robot")
    parser.add_argument("--agilex_port", default="", help="Serial/USB device path for AgileX follower")
    parser.add_argument("--agilex_robot_id", default="left", help="Follower arm identifier (left/right)")
    parser.add_argument(
        "--agilex_cameras_inline",
        default="",
        help="Inline YAML camera spec string (same as inference.py --cameras parameter)",
    )
    parser.add_argument(
        "--agilex_max_relative_target",
        default=None,
        type=int,
        help="Optional clipping magnitude forwarded to AlohaAgileXFollowerConfig",
    )
    parser.add_argument(
        "--agilex_use_degrees",
        action="store_true",
        help="Forward use_degrees flag to AlohaAgileXFollowerConfig",
    )
    parser.add_argument("--load_checkpoint_path",default="", help="Path to checkpoint to resume training from.", type=str)
    parser.add_argument("--start_step", default=0, help="Step to start training from.", type=int)
    parser.add_argument(
        "--reward_backtrack_steps",
        default=2,
        type=int,
        help="Number of previous action chunks to backtrack when assigning sub-task rewards. "
             "When user scores a sub-task at chunk i, reward is assigned to chunks [i-N, i].",
    )

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

    variant, args = parse_training_args(train_args_dict, parser)

    # 解析摄像头配置（与 inference.py 的 --cameras 参数格式完全一致）
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
    # Backwards compatibility: older modules expect `variant.instruction`.
    variant.instruction = getattr(variant, "task", "")
    print(variant)
    main(variant)
    sys.exit()
