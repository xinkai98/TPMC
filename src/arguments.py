import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--env_name", default="walker/walk")
    parser.add_argument("--domain_name", default=None)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--image_size", default=84, type=int)
    parser.add_argument("--image_crop_size", default=84, type=int)
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--eval_mode", default="all", type=str)
    parser.add_argument("--replay_buffer_capacity", default=500000, type=int)

    # agent
    parser.add_argument("--agent", default="sac", type=str)
    parser.add_argument("--num_env_steps", default="500100", type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--init_steps", default=1000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--auxiliary_update_freq", default=2, type=int)
    parser.add_argument("--num_train_updates", default=1, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)

    # actor
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)

    # critic
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)  # critic 的动量系数
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    parser.add_argument("--critic_weight_decay", default=0, type=float)

    # architecture
    parser.add_argument("--encoder_type", default='pixel', type=str)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_head_layers", default=0, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--encoder_feature_dim", default=50, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)  # encoder 的动量系数
    parser.add_argument("--encoder_lr", default=1e-3, type=float)

    # entropy maximization
    parser.add_argument("--init_temp", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)

    # auxiliary tasks
    parser.add_argument("--aux_lr", default=3e-4, type=float)
    parser.add_argument("--aux_beta", default=0.9, type=float)
    parser.add_argument("--aux_update_freq", default=2, type=int)


    # eval
    parser.add_argument("--eval_freq", default="10000", type=int)  # 每 10000 步 评估10次 共评估 50 次
    parser.add_argument("--num_eval_episodes", default=10, type=int)
    parser.add_argument("--distracting_cs_intensity", default=0.0, type=float)

    # misc
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--work_dir", default="./outputs", type=str)
    parser.add_argument("--save_video", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--save_buffer", default=False, action="store_true")
    parser.add_argument("--save_tb", default=True)
    parser.add_argument("--detach_encoder", default=False, action="store_true")

    parser.add_argument("--wandb", default=False, action="store_true")

    parser.add_argument("--log_interval", default=32, type=int)

    parser.add_argument("--mode", default=0, type=int)
    parser.add_argument("--step", default=500000, type=int)
    

    args = parser.parse_args()

    return args
