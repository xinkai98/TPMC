import contextlib
import torch
import numpy as np
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset
import time
import torchvision
from torchvision.utils import save_image, make_grid
import json
import augmentations
import torch.nn as nn

from captum.attr import GuidedBackprop, GuidedGradCam, LayerGradCam
# import torchvision.transforms.functional as F
import torch.nn.functional as nnf

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    with contextlib.suppress(OSError):
        os.mkdir(dir_path)
    return dir_path


def load_config(key=None):
    path = os.path.join('../setup', 'config.cfg')
    # path = os.path.join('setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    return data[key] if key is not None else data


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class HookFeatures:
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)

    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)

    def gradient_hook_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, action=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.action = action

    def forward(self, obs):
        if self.action is None:
            return self.model(obs)[0]
        return self.model(obs, self.action)[0]


def compute_guided_backprop(obs, action, model):  # 默认使用这个
    model = ModelWrapper(model, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    return attribution


def compute_guided_gradcam(obs, action, model):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(model, action=action)
    gbp = GuidedGradCam(model, layer=model.model.encoder.head_cnn.layers)
    attribution = gbp.attribute(obs, attribute_to_layer_input=True)
    return attribution
    

def compute_gradcam(obs, action, model):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(model, action=action)
    gbp = LayerGradCam(model, layer=model.model.encoder.head_cnn.layers)
    attribution = gbp.attribute(obs, attribute_to_layer_input=True, relu_attributions=True)
    return attribution


def compute_vanilla_grad(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    q, q2 = critic_target(obs, action.detach())
    q.sum().backward()
    return obs.grad


def compute_attribution(model, obs, action=None, method="guided_backprop"):
    if method == "guided_backprop":
        return compute_guided_backprop(obs, action, model)
    if method == 'guided_gradcam':
        return compute_guided_gradcam(obs, action, model)
    if method == 'gradcam':
        return compute_gradcam(obs, action, model)
    return compute_vanilla_grad(model, obs, action)


def compute_features_attribution(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    hook = HookFeatures(critic_target.encoder)
    q, _ = critic_target(obs, action.detach())
    q.sum().backward()
    features_gardients = hook.gradients
    hook.close()
    return obs.grad, features_gardients


def compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i: i + 3].abs().max(dim=1)[0]
        q = torch.quantile(attributions.flatten(1), quantile, 1)
        mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)


def make_obs_grid(obs, n=1):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            sample.append(obs[i, j: j + 3].unsqueeze(0))
    sample = torch.cat(sample, 0)
    return make_grid(sample, nrow=3) / 255.0


def make_attribution_pred_grid(attribution_pred, n=4):
    return make_grid(attribution_pred[:n], nrow=1)


def make_obs_grad_grid(obs_grad, n=1):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            channel_attribution, _ = torch.max(obs_grad[i, j: j + 3], dim=0)
            sample.append(channel_attribution[(None,) * 2] / channel_attribution.max())
    sample = torch.cat(sample, 0)
    q = torch.quantile(sample.flatten(1), 0.97, 1)
    sample[sample <= q[:, None, None, None]] = 0
    return make_grid(sample, nrow=3)


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            capacity,
            batch_size,  # 512
            auxiliary_task_batch_size,  # 128
            device,
            image_size=84,
            prefill=None,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.auxiliary_task_batch_size = auxiliary_task_batch_size
        self.device = device
        self.image_size = image_size

        # pixels obs are stored as uint8
        obs_dtype = np.uint8

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)

        # self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)  # shape:(100000, 9, 100, 100)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)  # for infinite bootstrap
        self.real_dones = np.empty((capacity, 1), dtype=np.float32)  # for auxiliary task

        self.idx = 0
        self.last_save = 0
        self.full = False

        self.current_auxiliary_batch_size = batch_size

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        # np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample_img_obs(self):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=self.batch_size)

        obses = self.obses[idxs]
        orig_obs = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = self.transform1(obses, self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        if self.transform2:
            obses = self.transform2(obses)

        # orig_obs = center_crop_image_batch(orig_obs, self.image_size)
        orig_obs = self.transform1(orig_obs, self.image_size)

        # print("after resize", orig_obs.shape, type(orig_obs))

        next_obses = self.transform1(next_obses, self.image_size)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        if self.transform2:
            next_obses = self.transform2(next_obses)

        pos = self.transform1(pos, self.image_size)
        pos = torch.as_tensor(pos, device=self.device).float()
        if self.transform2:
            pos = self.transform2(pos)

        # print('inside replay buffer', obses.shape, next_obses.shape)

        # visualize augmented images
        # if counter() <= 24:
        #    visualise_aug_obs(obses, self.transform.__name__)

        # obses = torch.as_tensor(obses, device=self.device).float()
        orig_obs = torch.as_tensor(orig_obs, device=self.device).float()
        # next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        # pos = torch.as_tensor(pos, device=self.device).float()
        info_dict = dict(obs_anchor=obses,
                         obs_pos=pos,
                         time_anchor=None,
                         time_pos=None)

        return orig_obs, obses, actions, rewards, next_obses, not_dones, info_dict

    def sample_RL_aug(self):
        """
        采样普通样本用于强化学习，并使用数据增强（随机裁剪）
        """
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = self.transform1(obses, self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        if self.transform2:
            obses = self.transform2(obses)

        next_obses = self.transform1(next_obses, self.image_size)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        if self.transform2:
            next_obses = self.transform2(next_obses)

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def sample_RL_aug_aux(self):
        """
        采样普通样本用于强化学习，并使用数据增强（随机裁剪），辅助任务也使用这些样本吧（？）
        """
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = self.transform1(obses, self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        if self.transform2:
            obses = self.transform2(obses)

        next_obses = self.transform1(next_obses, self.image_size)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        if self.transform2:
            next_obses = self.transform2(next_obses)

        pos = self.transform1(pos, self.image_size)
        pos = torch.as_tensor(pos, device=self.device).float()
        if self.transform2:
            pos = self.transform2(pos)

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones, pos

        # idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)
        #
        # obses = self.obses[idxs]
        # # orig_obs = self.obses[idxs]
        # next_obses = self.next_obses[idxs]
        # # pos = obses.copy()
        #
        # obses = self.transform1(obses, self.image_size)
        # obses = torch.as_tensor(obses, device=self.device).float()
        # # if self.transform2:
        # #     obses = self.transform2(obses)
        #
        # # orig_obs = center_crop_image_batch(orig_obs, self.image_size)
        # # orig_obs = self.transform1(orig_obs, self.image_size)
        #
        # # print("after resize", orig_obs.shape, type(orig_obs))
        #
        # next_obses = self.transform1(next_obses, self.image_size)
        # next_obses = torch.as_tensor(next_obses, device=self.device).float()
        # # if self.transform2:
        # #     next_obses = self.transform2(next_obses)
        #
        # # pos = self.transform1(pos, self.image_size)
        # # pos = torch.as_tensor(pos, device=self.device).float()
        # # if self.transform2:
        # #     pos = self.transform2(pos)
        #
        # # print('inside replay buffer', obses.shape, next_obses.shape)
        #
        # # visualize augmented images
        # # if counter() <= 24:
        # #    visualise_aug_obs(obses, self.transform.__name__)
        #
        # # obses = torch.as_tensor(obses, device=self.device).float()
        # # orig_obs = torch.as_tensor(orig_obs, device=self.device).float()
        # # next_obses = torch.as_tensor(next_obses, device=self.device).float()
        # actions = torch.as_tensor(self.actions[idxs], device=self.device)
        # rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        # not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        #
        # # pos = torch.as_tensor(pos, device=self.device).float()
        # # info_dict = dict(obs_anchor=obses,
        # #                  obs_pos=pos,
        # #                  time_anchor=None,
        # #                  time_pos=None)
        #
        # return obses, actions, rewards, next_obses, not_dones

    def sample_research(self):  # sample batch for auxiliary task
        """ 先采样训练辅助任务的样本 每个样本3个时间步的观测
            x_t x_t+1 x_t+2
        """
        idxs = np.random.randint(0,
                                 self.capacity - 3 if self.full else self.idx - 3,
                                 size=self.auxiliary_task_batch_size)
        idxs = idxs.reshape(-1, 1)
        step = np.arange(3).reshape(1, -1)  # this is a range 得到 [[0, 1, 2]]
        idxs = idxs + step  # 得到 auxiliary_task_batch_size 个开头的样本索引，每个样本包括 3 个索引
        """例如：array([[48, 49, 50],
                       [36, 37, 38],
                       [35, 36, 37],
                       [57, 58, 59],
                       [11, 12, 13],
                       [26, 27, 28]])
        """

        real_dones = torch.as_tensor(self.real_dones[idxs], device=self.device)  # (B, 3, 1)
        # we add this to avoid sampling the episode boundaries
        valid_idxs = torch.where((real_dones.mean(1) == 0).squeeze(-1))[0].cpu().numpy()
        idxs = idxs[valid_idxs]  # (B, 3)
        idxs = idxs[:self.auxiliary_task_batch_size] if idxs.shape[0] >= self.auxiliary_task_batch_size else idxs
        self.current_auxiliary_batch_size = idxs.shape[0]

        ogin_obses = self.obses[idxs]  # (B, 3, 3*3=9, 100, 100)
        ogin_obses = np.transpose(ogin_obses, (1, 0, 2, 3, 4))
        obses = []
        for i in range(3):
            obses.append(self.transform1(ogin_obses[i], self.image_size))
            if self.transform2:
                obses[i] = self.transform2(obses[i])
        obses = np.transpose(obses, (1, 0, 2, 3, 4))
        obses = torch.as_tensor(obses, device=self.device).float()

        # next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        # rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        # # not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)   # (B, jumps+1, 1)

        # au_samples  = {
        #     'observation': obses.transpose(0, 1).unsqueeze(3),
        #     'action': actions.transpose(0, 1),
        #     # 'reward': rewards.transpose(0, 1),
        # }  # spr_samples 为用于更新辅助任务的样本，其中 reward 没有用

        au_samples = dict(observation=obses,
                          action=actions)

        return (*self.sample_RL_aug(), au_samples)  # 调用 sample_RL_aug 方法采样正常用于训练 RL 的样本，并且使用数据增强

    def sample_mcc(self):  # sample batch for auxiliary task
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.auxiliary_task_batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = self.transform1(obses, self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        if self.transform2:
            obses = self.transform2(obses)

        next_obses = self.transform1(next_obses, self.image_size)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        if self.transform2:
            next_obses = self.transform2(next_obses)

        pos = self.transform1(pos, self.image_size)
        pos = torch.as_tensor(pos, device=self.device).float()
        if self.transform2:
            pos = self.transform2(pos)

        actions = torch.as_tensor(self.actions[idxs], device=self.device)

        au_samples = dict(obs_anchor=obses,
                          obs_pos=pos,
                          next_obs=next_obses,
                          action=actions)

        return (*self.sample_RL_aug(), au_samples)
        # return obses, actions, rewards, next_obses, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        pos = obs.copy()
        pos = torch.as_tensor(pos).cuda().float()

        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)
        pos = augmentations.random_shift(pos, pad)

        return obs, actions, rewards, next_obs, not_dones, pos


def save(self, save_dir):
    if self.idx == self.last_save:
        return
    path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
    payload = [
        self.obses[self.last_save: self.idx],
        self.next_obses[self.last_save: self.idx],
        self.actions[self.last_save: self.idx],
        self.rewards[self.last_save: self.idx],
        self.not_dones[self.last_save: self.idx],
    ]
    self.last_save = self.idx
    torch.save(payload, path)


def load(self, save_dir):
    chunks = os.listdir(save_dir)
    chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
    for chunk in chucks:
        start, end = [int(x) for x in chunk.split(".")[0].split("_")]
        path = os.path.join(save_dir, chunk)
        payload = torch.load(path)
        assert self.idx == start
        self.obses[start:end] = payload[0]
        self.next_obses[start:end] = payload[1]
        self.actions[start:end] = payload[2]
        self.rewards[start:end] = payload[3]
        self.not_dones[start:end] = payload[4]
        self.idx = end


def __getitem__(self, idx):
    idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
    idx = idx[0]
    obs = self.obses[idx]
    action = self.actions[idx]
    reward = self.rewards[idx]
    next_obs = self.next_obses[idx]
    not_done = self.not_dones[idx]

    if self.transform:
        obs = self.transform(obs)
        next_obs = self.transform(next_obs)

    return obs, action, reward, next_obs, not_done


def __len__(self):
    return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3:(i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f"{count:,}"


def visualise_aug_obs(obs, transform):
    # print("shape of obs inside visualise_aug_obs",obs.shape, obs[0].shape)=> (128, 9, 84, 84) (9, 84, 84)
    batch_tensor = obs[0].transpose(1, 2, 0)
    # print("shape after transpose", batch_tensor.shape), (84, 84, 9)
    batch_tensor = np.dsplit(batch_tensor, 3)
    # print('shape after dsplit', len(batch_tensor), batch_tensor[0].shape), 3 (84, 84, 3)
    out = [
        (batch_tensor[i].transpose(2, 0, 1)) / 255.0 for i in range(len(batch_tensor))
    ]
    # print('shape after transpose', len(out), out[0].shape), 3 (3, 84, 84)
    out = np.array(out)
    out = torch.from_numpy(out)
    grid_img = torchvision.utils.make_grid(out, nrow=3)
    # print("shape of grid", grid_img.shape), [3, 88, 260]
    save_image(grid_img, "aug_%s_%d.jpg" % (transform, counter()))


def counter():
    counter.counter = getattr(counter, "counter", 0) + 1
    return counter.counter


"""Config class"""


class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(
            self,
            env,
            replay_buffer,
            train,
            eval,
            critic,
            actor,
            encoder,
            decoder,
            sac,
            params,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.train = train
        self.eval = eval
        self.critic = critic
        self.actor = actor
        self.encoder = encoder
        self.decoder = decoder
        self.sac = sac
        self.params = params

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(
            params.env,
            params.replay_buffer,
            params.train,
            params.eval,
            params.critic,
            params.actor,
            params.encoder,
            params.decoder,
            params.sac,
            params.params,
        )


class HelperObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)
