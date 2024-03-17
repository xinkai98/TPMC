import numpy as np
import torch
import gym
import os
import time
from arguments import parse_args
# import dmc2gym
import wandb
import utils
from logger import Logger
from video import VideoRecorder

from sac import SacAgent
from tpmc_sac import TpmcAgent
from svea_sac import SVEAAgent
from sgqn_sac import SGQNAgent

from augmentations import center_crop_image, random_conv, random_crop, random_overlay
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
               ) + args.encoder_type

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
    if args.agent == "tpmc_sac":
        return TpmcAgent(obs_shape, action_shape, device, args)
    if args.agent == "svea_sac":
        return SVEAAgent(obs_shape, action_shape, device, args) 
    if args.agent == "sgqn_sac":
        return SGQNAgent(obs_shape, action_shape, device, args)
    else:
        assert f"agent is not supported: {args.agent}"


def evaluate(env, agent, video, num_episodes, L, step, args, test_env=False, eval_mode=None):
    episode_rewards = []
    start_time = time.time()
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        episode_step = 0

        _test_env = f"_test_env_{eval_mode}" if test_env else ""

        while not done:
            with torch.no_grad():
                if args.encoder_type == "pixel":
                    obs = transforms["center_crop_image"](obs, args.image_size)
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
                episode_step += 1

        video.save(f"{step}_{_test_env}.mp4")
        L.log(f"eval/episode_reward{_test_env}", episode_reward, step)

        episode_rewards.append(episode_reward)

    L.log(f'eval/eval_time{_test_env}', time.time() - start_time, step)
    mean_ep_reward = np.mean(episode_rewards)
    L.log(f'eval/mean_episode_reward{_test_env}', mean_ep_reward, step)


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
                   image_size=args.image_size,)

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
                mode=eval_mode,)
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

    

    replay_buffer = utils.ReplayBuffer(obs_shape=env.observation_space.shape, action_shape=action_shape, capacity=args.replay_buffer_capacity,
                                           batch_size=args.batch_size, auxiliary_task_batch_size=args.batch_size,  # 不用这个
                                           device=device, image_size=args.image_size,  # 84
                                           )
        
   
    agent = make_agent(obs_shape=crooped_obs_shape,
                       action_shape=action_shape,
                       device=device,
                       args=args,)

    L = Logger(args.work_dir, use_tb=args.save_tb)

    print("----------------------")
    print("details of agent-")
    print(agent)
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    # mean_all_episode_rewards = []
    for step in range(0, args.num_env_steps, args.action_repeat):
        # evaluate agent periodically

        if step % args.eval_freq == 0 :
            print("Evaluating:", args.work_dir)
            L.log('eval/episode', episode, step)

            evaluate(env, agent, video, args.num_eval_episodes, L, step, args)

            if test_envs is not None:
                for test_env, test_env_mode in zip(test_envs, test_envs_mode):
                    evaluate(test_env, agent, video, args.num_eval_episodes, L, step, args,
                             test_env=True,
                             eval_mode=test_env_mode, )

            L.dump(step)

            if step % 100000 == 0 and args.save_model:
                agent.save(model_dir, step)

            # if args.save_buffer:
            #     replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                L.dump(step)

                start_time = time.time()

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step, args.wandb)

        next_obs, reward, done, _ = env.step(action)

        # allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        episode_reward += reward
        obs = next_obs
        episode_step += 1


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    # torch.backends.cudnn.benchmark = False
    args = parse_args()
    main(args)


