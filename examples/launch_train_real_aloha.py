import argparse
import sys

from examples.train_real_aloha import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, help="Random seed.", type=int)
    parser.add_argument("--launch_group_id", default="", help="group id used to group runs on wandb.")
    parser.add_argument("--eval_episodes", default=10, help="Number of episodes used for evaluation.", type=int)
    parser.add_argument("--env", default="aloha", help="Name of environment")
    parser.add_argument("--log_interval", default=1000, help="Logging interval.", type=int)
    parser.add_argument("--eval_interval", default=5000, help="Eval interval.", type=int)
    parser.add_argument("--checkpoint_interval", default=-1, help="Checkpoint interval.", type=int)
    parser.add_argument("--batch_size", default=64, help="Mini batch size.", type=int)
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
        "--instruction",
        default="put the spoon on the plate",
        help="language instruction for the robot",
    )
    parser.add_argument("--proprio_dim", default=14, help="dimension of proprioceptive state", type=int)
    parser.add_argument("--img_feature_dim", default=2024, help="dimension of pi0 visual features", type=int)
    parser.add_argument("--arm_dof", default=6, help="Number of joints per arm", type=int)
    parser.add_argument(
        "--image_order",
        nargs="+",
        default=["high", "low", "left_wrist", "right_wrist"],
        help="ordering of cameras for pixel observations",
    )
    parser.add_argument(
        "--external_camera",
        default="high",
        help="camera alias used as exterior view (see image_order entries)",
    )
    parser.add_argument(
        "--wrist_camera",
        default="left_wrist",
        help="camera alias used as wrist input for pi0",
    )
    parser.add_argument(
        "--gripper_indices",
        nargs="+",
        default=[-2, -1],
        type=int,
        help="indices inside the pi0 action vector corresponding to grippers",
    )
    parser.add_argument(
        "--real_env_max_steps",
        default=1000,
        type=int,
        help="maximum number of low-level control steps per real rollout",
    )
    parser.add_argument("--control_hz", default=15, type=int, help="command frequency for the robot")

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
    print(variant)
    main(variant)
    sys.exit()
