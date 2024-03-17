import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
from feature_extractor import Encoder, SharedCNN, HeadCNN, RLProjection
import wandb
import augmentations as augmentations
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


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


class Actor(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )
        self.mlp.apply(weight_init)

    def forward(
        self,
        obs,
        compute_pi=True,
        compute_log_pi=True,
        detach_encoder=False,
        compute_attrib=False,
    ):
        obs = self.encoder(obs, detach_encoder)
        mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std, obs


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))


class Critic(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.Q1 = QFunction(self.encoder.out_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.out_dim, action_shape[0], hidden_dim)

    def forward(self, x, action, detach_encoder=False):
        x = self.encoder(x, detach_encoder)
        return self.Q1(x, action), self.Q2(x, action)


class SacAgent(object):
    """ SAC."""

    def __init__(self, obs_shape, action_shape, device, args):

        self.device = device
        self.image_size = obs_shape[-1]

        self.detach_encoder = args.detach_encoder  # False
        self.hidden_dim = args.hidden_dim  # 1024
        self.discount = args.discount  # 0.99
        self.init_temp = args.init_temp  # 0.1
        self.alpha_lr = args.alpha_lr  # 温度系数学习率 1e-4
        self.alpha_beta = args.alpha_beta  # 0.5
        self.actor_lr = args.actor_lr  # 1e-3
        self.actor_beta = args.actor_beta  # 0.9
        self.actor_log_std_min = args.actor_log_std_min  # -10
        self.actor_log_std_max = args.actor_log_std_max  # 2
        self.actor_update_freq = args.actor_update_freq  # 2
        self.critic_lr = args.critic_lr  # 1e-3
        self.critic_beta = args.critic_beta  # 0.9
        self.critic_tau = args.critic_tau  # 0.1
        self.critic_target_update_freq = args.critic_target_update_freq  # 2

        # self.encoder_type = "pixel"
        self.encoder_feature_dim = args.encoder_feature_dim  # 50
        self.encoder_lr = args.encoder_lr  # 1e-3
        self.encoder_tau = args.encoder_tau  # 0.05
        self.num_layers = args.num_layers  # 4
        self.num_filters = args.num_filters  # 32
        self.log_interval = args.log_interval  # 32

        # 共享卷积层
        shared_cnn = SharedCNN(obs_shape, self.num_layers, self.num_filters).to(device)
        # 此网络只包含 flatten 操作，额外卷积层 层数为 0
        head_cnn = HeadCNN(shared_cnn.out_dim, 0, self.num_filters).to(device)
        # 计算得到 全连接层 的输入层维度：卷积核数目 * 特征图长 * 特征图宽
        projection_in_dim = self.num_filters * head_cnn.out_dim * head_cnn.out_dim

        # AC网络的编码器，其中包含共享参数的 卷积层 和 独立参数的 全连接层
        actor_encoder = Encoder(shared_cnn, head_cnn,
                                RLProjection(projection_in_dim, self.encoder_feature_dim))

        critic_encoder = Encoder(shared_cnn, head_cnn,
                                 RLProjection(projection_in_dim, self.encoder_feature_dim))

        # 利用上述定义的 AC 编码器分别 新建 AC 网络
        self.actor = Actor(actor_encoder, action_shape, self.hidden_dim,
                           self.actor_log_std_min, self.actor_log_std_max).to(device)

        self.critic = Critic(critic_encoder, action_shape, self.hidden_dim).to(device)

        # 利用深复制新建 target critic 网络
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(self.init_temp)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr,
                                                betas=(self.actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 betas=(self.critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.alpha_lr,
                                                    betas=(self.alpha_beta, 0.999))

        self.train()
        self.critic_target.train()

        tb_dir = os.path.join(
            args.work_dir,
            "tensorboard",
        )
        self.writer = SummaryWriter(tb_dir)

    """function to help toggle between train and eval mode"""

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        # _obs = self._obs_to_input(obs)
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = augmentations.center_crop_image(obs, self.image_size)
        # print("inside sample action", obs.shape)
        # _obs = self._obs_to_input(obs)
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            # print("inside sample action torch.no grad", obs.shape)
            _, pi, _, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step, WB_LOG):
        with torch.no_grad():
            _, policy_action, log_pi, _, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
        # set detach_encoder to True to stop critic's gradient flow to the encoder.

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(L, step, WB_LOG)

    def update_actor_and_alpha(self, obs, L, step, WB_LOG):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std, _ = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)


        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step, WB_LOG):

        obs, action, reward, next_obs, not_done, _ = replay_buffer.sample_drq()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step, WB_LOG)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, WB_LOG)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder,
                                     self.encoder_tau)
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     self.critic_tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), f"{model_dir}/actor_{step}.pt")
        torch.save(self.critic.state_dict(), f"{model_dir}/critic_{step}.pt")

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load(f"{model_dir}/actor_{step}.pt", map_location='cuda'))
        self.critic.load_state_dict(torch.load(f"{model_dir}/critic_{step}.pt", map_location='cuda'))
