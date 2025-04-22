from distutils.util import strtobool

import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from wrappers import build_env
import numpy as np
import argparse

from TPSgym.test_env import TestEnv
from TPSgym.easy_env import EasyEnv
from TPSgym.shoot_env import ShootEnv
from TPSgym.ma_env import MaEnv
import gym
from gym import spaces
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models
import datetime


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
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(7616, 512)),
            nn.ReLU(),
        )
        self.map_network = nn.Sequential(
            layer_init(nn.Conv2d(2, 32, 8, stride=4)),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(7616, 512)),
            nn.ReLU(),
        )
        self.actor1 = nn.Sequential(
            # layer_init(nn.Linear(512, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(512, env.action_space.nvec[0]), std=0.01),
        )
        self.actor2 = nn.Sequential(
            # layer_init(nn.Linear(512, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(512, env.action_space.nvec[1]), std=0.01),
        )
        self.actor3 = nn.Sequential(
            # layer_init(nn.Linear(512, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(512, env.action_space.nvec[2]), std=0.01),
        )
        self.critic = nn.Sequential(
            # layer_init(nn.Linear(512, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1),
        )
        self.game_info = nn.Sequential(
            layer_init(nn.Linear(22, 64), std=0.01),
            nn.ReLU(),
            # layer_init(nn.Linear(32, 32), std=0.01),
            # nn.ReLU(),
        )
        self.linear = nn.Sequential(
            layer_init(nn.Linear(64 + 512 + 512, 512), std=0.01),
            nn.ReLU(),
        )
        self.find = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=0.01),
        )
        self.rnn = layer_init(nn.LSTM(512, 512, batch_first=True))
        self.rnn_hidden = None

    def get_value(self, x, detect):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        x = self.network(x / 255.0)
        detect = self.detect(detect)
        detect = detect.squeeze(1)
        x = torch.cat((x, detect), dim=1)
        x = self.linear(x)
        # x, self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        return self.critic(x)

    def get_action_and_value(self, x, g_info, detect_m, action=None):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
            detect_m = detect_m.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1)
            detect_m = detect_m.permute(2, 0, 1)
        action_mask1 = torch.zeros(g_info.shape[0], env.action_space.nvec[0]).to(device)
        action_mask2 = torch.zeros(g_info.shape[0], env.action_space.nvec[1]).to(device)
        action_mask3 = torch.zeros(g_info.shape[0], env.action_space.nvec[2]).to(device)
        action_mask1[:, 0] = 1
        # action_mask[:, 9] = 1
        g_info = g_info.squeeze(1)
        for i in range(g_info.shape[0]):
            if g_info[i, -1] < 0.5:
                action_mask3[i, 1] = 1
            else:
                if g_info[i, 17] < 0.65:
                    action_mask2[i, 9] = 1  # no turn right
                if g_info[i, 17] > 0.35:
                    action_mask2[i, 10] = 1  # no turn left

        hidden = self.network(x / 255.0)
        find = self.find(hidden)
        g_info = self.game_info(g_info)
        detect_m = self.map_network(detect_m)

        hidden = torch.cat((hidden, g_info, detect_m), dim=1)
        hidden = self.linear(hidden)
        hidden, self.rnn_hidden = self.rnn(hidden, self.rnn_hidden)

        logits1 = self.actor1(hidden)
        logits2 = self.actor2(hidden)
        logits3 = self.actor3(hidden)

        logits1 = logits1 - action_mask1 * 1e8
        logits2 = logits2 - action_mask2 * 1e8
        logits3 = logits3 - action_mask3 * 1e8
        probs1 = Categorical(logits=logits1)
        probs2 = Categorical(logits=logits2)
        probs3 = Categorical(logits=logits3)
        if action is None:
            action1 = probs1.sample()
            action2 = probs2.sample()
            action3 = probs3.sample()
            action = torch.stack((action1, action2, action3), dim=1).squeeze(0)
        else:
            action1 = action[:, 0]
            action2 = action[:, 1]
            action3 = action[:, 2]
        return action, probs1.log_prob(action1) + probs2.log_prob(action2) + probs3.log_prob(
            action3), probs1.entropy() + probs2.entropy() + probs3.entropy(), self.critic(hidden), find

    def reset_rnn_hidden(self):
        self.rnn_hidden = None


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
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
    parser.add_argument("--env-id", type=str, default="Easy_Env-v1",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128 * 2,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
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
    parser.add_argument("--max-grad-norm", type=float, default=0.3,
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

    seed_it(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    env = MaEnv(image_shape=(84, 168, 1))
    agent = Agent().to(device)
    # loss_fd = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    initial_val = None
    while True:
        print("waiting for UE..")
        initial_val = env.reset()
        env.reset_connection()
        if initial_val is not None:
            break
        time.sleep(1)
    load_model = True
    model_load_path = 'outputs/data_save/ppo_ma_model.pt'
    if load_model and os.path.exists(model_load_path):
        checkpoint = torch.load(model_load_path)
        agent.actor1.load_state_dict(checkpoint['actor1'])
        agent.actor2.load_state_dict(checkpoint['actor2'])
        agent.actor3.load_state_dict(checkpoint['actor3'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.network.load_state_dict(checkpoint['network'])
        agent.map_network.load_state_dict(checkpoint['map_network'])
        agent.find.load_state_dict(checkpoint['find'])
        agent.game_info.load_state_dict(checkpoint['game_info'])
        agent.linear.load_state_dict(checkpoint['linear'])
        agent.rnn.load_state_dict(checkpoint['rnn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Model loaded from {}'.format(model_load_path))

    obs = torch.zeros((args.num_steps*4, args.num_envs) + env.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps*4, args.num_envs) + env.action_space.shape).to(device)
    logprobs = torch.ones((args.num_steps*4, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps*4, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps*4, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps*4, args.num_envs)).to(device)

    game_info = torch.zeros((args.num_steps*4, args.num_envs) + (1, 22)).to(device)
    detect_map = torch.zeros((args.num_steps*4, args.num_envs) + (84, 168, 2)).to(device)
    tmp = env.reset()
    if tmp is None:
        # fail, reset env
        while True:
            print("UE broken.. wait for UE restarting")
            env.reset_connection()
            if env.test_ue():
                break
            time.sleep(5)
        # ue recovery, reset env
        while True:
            if env.reset() is not None:
                break
            time.sleep(1)
        next_obs, _, _, _ = env.step(np.zeros(4))
    else:
        next_obs = tmp
    next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    next_game_info = env.game_info
    next_detect_map = env.detect_map.unsqueeze(0)

    data_save_path = 'outputs/data_save'
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    for iter in range(args.num_steps*4):
        obs[iter] = next_obs
        dones[iter] = next_done
        detect_map[iter] = next_detect_map
        game_info[iter] = next_game_info

        with torch.no_grad():
            action, logprob, _, value, find = agent.get_action_and_value(next_obs, next_game_info, next_detect_map)
        actions[iter] = action
        # logprobs[iter] = logprob
        # values[iter] = value.squeeze(1)

        next_obs, reward, done, info = env.step(action.cpu().numpy())
        next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
        # next_done = torch.Tensor(done).to(device)
        next_game_info = torch.Tensor(env.game_info).unsqueeze(0).to(device)
        next_detect_map = env.detect_map.unsqueeze(0).to(device)

        if info.get('game over', False):
            next_obs = torch.Tensor(env.reset()).unsqueeze(0).to(device)
            next_game_info = torch.Tensor(env.game_info).unsqueeze(0).to(device)
            next_detect_map = env.detect_map.unsqueeze(0).to(device)

    b_obs = obs.reshape((-1,) + env.observation_space.shape)
    # b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + env.action_space.shape)
    # b_advantages = advantages.reshape(-1)
    # b_returns = returns.reshape(-1)
    # b_values = values.reshape(-1)
    b_game_info = game_info.reshape((-1,) + (1, 22))
    b_detect_map = detect_map.reshape((-1,) + (84, 168, 2))
    # torch.save({'obs': b_obs, 'actions': b_actions, 'game_info': game_info, 'detect_map':detect_map}, data_save_path+'/ppo_data.pt')
