from distutils.util import strtobool

import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from wrappers import build_env
import numpy as np
import argparse

from TPSgym.test_env import TestEnv
from TPSgym.shoot_env import ShootEnv
import gym
from gym import spaces
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, bias_const)
        elif 'weight' in name:
            nn.init.orthogonal_(param, std)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(7616, 512)),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            # layer_init(nn.Linear(512, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(512, env.action_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            # layer_init(nn.Linear(512, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1),
        )
        # self.actor = layer_init(nn.Linear(512, env.action_space.n), std=0.01)
        # self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        x = self.network(x / 255.0)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if x.dim() == 3:
            x = x.permute(2, 0, 1)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None: # choose the max action
            # action = logits.argmax(dim=-1)
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Test_Env-v0",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128*4,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def seed_it(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)




if __name__ == "__main__":
    args = parse_args()
    model_load_path = 'outputs/ppo_model.pt'
    load_model = True
    seed_it(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    env = TestEnv(image_shape=(84, 168, 1))
    agent = Agent().to(device)
    if load_model and os.path.exists(model_load_path):
        checkpoint = torch.load(model_load_path)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.network.load_state_dict(checkpoint['network'])

    nepisode = 0
    best_reward = -np.inf
    rewards_episode = []
    time_start = time.time()
    next_obs = torch.Tensor(env.reset()).unsqueeze(0).to(device)
    for step in range(0, args.total_timesteps):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(action.cpu().numpy())
        # if reward:
        #     print("reward", reward)
        next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
        if info.get('game over', False):
            nepisode += 1
            reward_, length = env.reward, env.game_cnt
            time_end = time.time()
            time_cost = time_end - time_start
            print(
                'Time steps so far: {}, episode so far: {}, '
                'episode reward: {:.4f}, episode length: {}, '
                'time cost: {:.2f} s'.format(step, nepisode, reward_, length, time_cost)
            )
        if done:
            next_obs = torch.Tensor(env.reset()).unsqueeze(0).to(device)

