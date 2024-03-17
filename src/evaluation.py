import numpy as np
import torch
import gym
import os
import time
import random
from arguments import parse_args
# import dmc2gym
import wandb
import utils
from logger import Logger
from video import VideoRecorder

from utils import (
    compute_attribution,
    compute_attribution_mask,
    make_obs_grid,
    make_obs_grad_grid,
)

from sac import SacAgent
from tpmc_sac import MccAgent
from svea_sac import SVEAAgent
from sgqn_sac import SGQNAgent

from augmentations import center_crop_image, random_conv, random_crop, random_overlay
from visualize import visualize_tsne
from collections import Counter
from env.wrappers import make_env

transforms = {
    "random_crop": random_crop,
    # "random_shift": random_shift,
    "random_conv": random_conv,
    "center_crop_image": center_crop_image,
    "random_overlay": random_overlay,
    "None": None
}


def make_directory(args):
    ts = time.gmtime()
    ts = time.strftime("%m%d", ts)
    env_name = f"{args.domain_name}-{args.task_name}"
    exp_name = (
                       (
                               (
                                       f"{env_name}-{args.agent}-{ts}-im{str(args.image_size)}-b"
                                       + str(args.batch_size)
                               )
                               + "-s"
                       )
                       + str(args.seed)
                       + "-"
               ) + args.encoder_type + str(args.mode) + "video" + str(args.step)

    args.work_dir = f"{args.work_dir}/{args.agent}/{exp_name}"

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, "video"))
    model_dir = utils.make_dir(os.path.join(args.work_dir, "model"))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, "buffer"))
    aug_dir = utils.make_dir(os.path.join(args.work_dir, "augmentated_obs"))

    return video_dir, model_dir, buffer_dir, aug_dir


def make_agent(obs_shape, action_shape, device, args):
    if args.agent == "sac":
        return SacAgent(obs_shape, action_shape, device, args)
    if args.agent == "mcc_sac":
        return MccAgent(obs_shape, action_shape, device, args)
    if args.agent == "svea_sac":
        return SVEAAgent(obs_shape, action_shape, device, args)
    if args.agent == "sgqn_sac":
        return SGQNAgent(obs_shape, action_shape, device, args)
    else:
        assert f"agent is not supported: {args.agent}"


def evaluate(env, agent, video, num_episodes, L, step, args, device, test_env=False, eval_mode=None):
    obs = env.reset()
    video.init()
    done = False
    torch_obs = []
    torch_action = []
    episode_step = 0

    _test_env = f"_test_env_{eval_mode}" if test_env else ""

    while not done:
        with torch.no_grad():
            obs = transforms["center_crop_image"](obs, args.image_size)
            with utils.eval_mode(agent):
                action = agent.select_action(obs)

                if episode_step == 0:
                    _obs = agent._obs_to_input(obs)
                    torch_obs.append(_obs)
                    torch_action.append(
                        torch.tensor(action).to(_obs.device).unsqueeze(0)
                    )
                    prefix = "eval" if eval_mode is None else eval_mode
                    if episode_step == 0:
                        obs_ = torch.cat(torch_obs, 0)
                        print(obs_.shape)
                        original_obs = make_obs_grid(obs_)
                        print(original_obs.shape)
                        agent.writer.add_image(
                            prefix + f"/original_{step}", original_obs, global_step=step
                        )

                        action_ = torch.cat(torch_action, 0)

                        obs_grad = compute_attribution(agent.critic, obs_, action_.detach())
                        grad_grid = make_obs_grad_grid(obs_grad.data.abs())
                        agent.writer.add_image(
                            prefix + f"/attributions_{step}", grad_grid, global_step=step
                        )

                        mask = compute_attribution_mask(obs_grad, quantile=0.95)
                        masked_obs = make_obs_grid(obs_ * mask)
                        agent.writer.add_image(
                            prefix + f"/masked_obs_{step}", masked_obs, global_step=step
                        )

                        masked_obs_1 = obs_ * mask
                        masked_obs_1[mask < 1] = random.uniform(-255., 255.)
                        masked_obs_1 = make_obs_grid(masked_obs_1)
                        agent.writer.add_image(
                            prefix + f"/masked_unified_1{step}", masked_obs_1, global_step=step
                        )

                        masked_obs_2 = obs_ * mask
                        masked_obs_2[mask < 1] = random.uniform(-255., 255.)
                        masked_obs_2 = make_obs_grid(masked_obs_2)
                        agent.writer.add_image(
                            prefix + f"/masked_unified_2{step}", masked_obs_2, global_step=step
                        )

                        masked_obs_3 = obs_ * mask
                        masked_obs_3[mask < 1] = random.uniform(-255., 255.)
                        masked_obs_3 = make_obs_grid(masked_obs_3)
                        agent.writer.add_image(
                            prefix + f"/masked_unified_3{step}", masked_obs_3, global_step=step
                        )

                        # attrib_grid = make_obs_grad_grid(torch.sigmoid(mask))
                        # agent.writer.add_image(
                        #     prefix + f"/smooth_attrib_{step}", attrib_grid, global_step=step
                        # )

                obs, reward, done, _ = env.step(action)
                video.record(env)

            episode_step += 1

        video.save(f"{step}_{_test_env}.mp4")

def main(args):
    # home = os.environ["HOME"]
    # os.environ["MJKEY_PATH"] = f"{home}/.mujoco/mujoco200_linux/bin/mjkey.txt"
    # os.environ["MUJOCO_GL"] = "egl"

    args.init_steps *= args.action_repeat
    args.log_interval *= args.action_repeat
    args.actor_update_freq *= args.action_repeat
    args.critic_target_update_freq *= args.action_repeat

    args.seed = args.seed
    args.gpuid = args.gpuid
    args.domain_name = args.domain_name or args.env_name.split('/')[0]
    args.task_name = args.task_name or args.env_name.split('/')[1]
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    print("random seed value", args.seed)
    torch.cuda.set_device(args.gpuid)
    utils.set_seed_everywhere(args.seed)

    gym.logger.set_level(40)
    # https://stackoverflow.com/questions/48605843/getting-a-strange-output-when-using-openai-gym-render
    # to hide the warnings. This will set minimal level of logger message to be printed to 40.
    # Correspondingly, only error level messages will be displayed.
    # You will need to include this statement every time you import gym

    # if args.transform2 == "random_conv":
    #     pre_transform_image_size = args.image_size
    # else:
    #     pre_transform_image_size = args.pre_transform_image_size

    env = make_env(domain_name=args.domain_name,
                   task_name=args.task_name,
                   seed=args.seed,
                   episode_length=args.episode_length,
                   frame_stack=3,
                   action_repeat=args.action_repeat,
                   image_size=args.image_size, )

    env.seed(args.seed)

    if args.eval_mode is not None:
        test_envs = []
        test_envs_mode = []

    if args.eval_mode in [
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
        "distracting_cs",
    ]:
        test_env = make_env(domain_name=args.domain_name,
                            task_name=args.task_name,
                            seed=args.seed + 42,
                            episode_length=args.episode_length,
                            action_repeat=args.action_repeat,
                            image_size=args.image_size,
                            mode=args.eval_mode)

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

    if args.eval_mode == "all":
        for eval_mode in ["video_easy", "video_hard"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode=eval_mode, )
            test_envs.append(test_env)
            test_envs_mode.append(eval_mode)

    action_shape = env.action_space.shape

    crooped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )

    '''
    # stack several consecutive frames together
    if config.encoder.type == "pixel":
        env = utils.FrameStack(env, k=config.env.frame_stack)
        test_env = utils.FrameStack(env_eval, k=config.env.frame_stack)
    '''

    video_dir, model_dir, buffer_dir, aug_dir = make_directory(args)

    video = VideoRecorder(video_dir if args.save_video else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = make_agent(obs_shape=crooped_obs_shape,  # RL智能体接收的是裁剪后的
                       action_shape=action_shape,
                       device=device,
                       args=args, )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    print("----------------------")
    print("details of agent-")
    print(agent)

    test_dir = f"model/{args.domain_name}_{args.task_name}/{args.agent}"

    step = args.step
    agent.load(test_dir, step)
    evaluate(env, agent, video, args.num_eval_episodes, L, step, args, device=device)
    if test_envs is not None:
        for test_env, test_env_mode in zip(test_envs, test_envs_mode):
            evaluate(test_env, agent, video, args.num_eval_episodes, L, step, args, device=device,
                     test_env=True,
                     eval_mode=test_env_mode)

        L.dump(step)

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    # torch.backends.cudnn.benchmark = False
    args = parse_args()
    main(args)



