``` python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import utils
from .critic import DoubleQCritic
from .actor import DiagGaussianActor

critic_cfg = {'hidden_dim': 256, 'hidden_depth': 2}
actor_cfg = {'hidden_dim': 256, 'hidden_depth': 2, 'log_std_bounds': [-20, 2]}


class SACAgent:
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        # critic와 critic_target을 직접 생성
        self.critic = DoubleQCritic(
            obs_dim, action_dim,
            critic_cfg['hidden_dim'],
            critic_cfg['hidden_depth']
        ).to(self.device)
        self.critic_target = DoubleQCritic(
            obs_dim, action_dim,
            critic_cfg['hidden_dim'],
            critic_cfg['hidden_depth']
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # actor 객체를 직접 생성
        self.actor = DiagGaussianActor(
            obs_dim, action_dim,
            actor_cfg['hidden_dim'],
            actor_cfg['hidden_depth'],
            actor_cfg['log_std_bounds']
        ).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # target entropy: action 차원 수의 음수
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                  lr=actor_lr,
                                                  betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        # 입력이 이미 텐서라면 device를 맞추고, 아니라면 새로 생성
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)
        # 배치 입력으로 처리 (unsqueeze 제거)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        # 배치 형태임을 확인 (예: (batch_size, action_dim))
        assert action.ndim == 2
        # 배치 전체 반환 (두 번째 반환값은 예시로 None)
        return utils.to_np(action), None

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # 현재 Q값 계산
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
        self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

```