import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
from transition_model import make_transition_model
import augmentations as augmentations
import random
from sac import SacAgent

from utils import (
    compute_attribution,
    compute_attribution_mask,
)


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


class Tpmc(nn.Module):
    def __init__(self, z_dim,
                 encoder,
                 encoder_target,
                 action_shape,
                 output_type="continuous"):
        super(Tpmc, self).__init__()

        self.encoder = encoder
        self.encoder_target = encoder_target

        self.transition_model = make_transition_model('deterministic', z_dim, action_shape, 512)
        self.transition_model_target = make_transition_model('deterministic', z_dim, action_shape, 512)

        for param_transition, param_transition_target in zip(self.transition_model.parameters(),
                                                             self.transition_model_target.parameters()):
            param_transition_target.data.copy_(param_transition.data)  # initialize
            param_transition_target.requires_grad = False

        self.W1 = nn.Parameter(torch.rand(z_dim, z_dim))

        # self.W2 = nn.Parameter(torch.rand(self.action_latent_dim + z_dim, z_dim))

        self.output_type = output_type  # output_type="continuous"

        self.apply(weight_init)

    def encode(self, obs, action, next_obs, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                state_p = self.encoder_target(obs)
                nextstate_p = self.transition_model_target.sample_prediction(torch.cat([state_p, action], dim=1))
                next_state_true = self.encoder_target(next_obs)

                if detach:
                    nextstate_p = nextstate_p.detach()
                    next_state_true = next_state_true.detach()
                return nextstate_p, next_state_true

        else:
            state_anchor = self.encoder(obs)
            nextstate_anchor = self.transition_model.sample_prediction(torch.cat([state_anchor, action], dim=1))

            if detach:
                nextstate_anchor = nextstate_anchor.detach()


            return nextstate_anchor

    def compute_logits(self, z_a, z_pos, equ=True):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        if equ:
            Wz = torch.matmul(self.W1, z_pos.T)  # (z_dim,B)
            logits = torch.matmul(z_a, Wz)  # (B,B)
            logits = logits - torch.max(logits, 1)[0][:, None]
        else:
            Wz = torch.matmul(self.W2, z_pos.T)  # (z_dim,B)
            logits = torch.matmul(z_a, Wz)  # (B,B)
            logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class TpmcAgent(SacAgent):

    def __init__(self, obs_shape, action_shape, device, args):

        super().__init__(obs_shape, action_shape, device, args)

        self.auxiliary_update_freq = args.auxiliary_update_freq  # 2
        self.action_repeat = args.action_repeat

        self.quantile = 0.95

        self.transition_model_tau = 0.005
        self.aux_lr = args.aux_lr

        self.Tpmc = Tpmc(self.encoder_feature_dim, self.critic.encoder,
                         self.critic_target.encoder, action_shape, output_type='continuous').to(self.device)

        self.Tpmc_optimizer = torch.optim.Adam(self.Tpmc.parameters(), lr=self.aux_lr, betas=(args.aux_beta, 0.999))

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None, WB_LOG=None):
        with torch.no_grad():
            _, policy_action, log_pi, _, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        obs_grad = compute_attribution(self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, self.quantile)
        masked_obs = obs * mask

        masked_obs[mask < 1] = random.uniform(obs.view(-1).min(), obs.view(-1).max())
        
        masked_Q1, masked_Q2 = self.critic(masked_obs, action)
        # critic_loss += 0.5 * (F.mse_loss(current_Q1, masked_Q1) + F.mse_loss(current_Q2, masked_Q2))
        critic_loss += 0.5 * (F.mse_loss(masked_Q1, target_Q) + F.mse_loss(masked_Q2, target_Q))

        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_aux(self, obs, action, next_obs, L, step, pos):


        # obs_grad = compute_attribution(self.critic, obs, action.detach())
        # mask = compute_attribution_mask(obs_grad, quantile=self.quantile)

        s_tilde = augmentations.random_overlay(obs.clone())
        # obs = obs * mask + s_tilde * (~mask)
        obs = s_tilde

        nextstate_anchor = self.Tpmc.encode(obs, action, next_obs)
        nextstate_pos, nextstate_true = self.Tpmc.encode(pos, action, next_obs, ema=True, detach=True)

        logits_1 = self.Tpmc.compute_logits(nextstate_anchor, nextstate_pos)
        labels_1 = torch.arange(logits_1.shape[0]).long().to(self.device)
        mvc_loss = self.cross_entropy_loss(logits_1, labels_1)

        # pred_loss = self.mse_loss(nextstate_anchor, nextstate_true)
        h0 = F.normalize(nextstate_anchor, p=2, dim=1)
        h1 = F.normalize(nextstate_true, p=2, dim=1)
        pred_loss = F.mse_loss(h0, h1)

        # loss = mvc_loss
        loss = 0.5 * mvc_loss + 0.5 * pred_loss
        # loss = pred_loss

        self.Tpmc_optimizer.zero_grad()

        loss.backward()

        self.Tpmc_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/mvc_loss', mvc_loss, step)
            L.log('train/pred_loss', pred_loss, step)
            L.log('train/total_loss', loss, step)


    def update(self, replay_buffer, L, step, WB_LOG):

        obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_drq()  # 随机 shift

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
            utils.soft_update_params(
                self.Tpmc.transition_model, self.Tpmc.transition_model_target, self.transition_model_tau
            )

        if step & self.auxiliary_update_freq == 0:
            self.update_aux(obs, action, next_obs, L, step, pos)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(), f"{model_dir}/actor_{step}.pt")
        torch.save(self.critic.state_dict(), f"{model_dir}/critic_{step}.pt")

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load(f"{model_dir}/actor_{step}.pt", map_location='cuda'))
        self.critic.load_state_dict(torch.load(f"{model_dir}/critic_{step}.pt", map_location='cuda'))
