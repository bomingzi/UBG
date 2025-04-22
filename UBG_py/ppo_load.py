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
        self.detect = nn.Sequential(
            layer_init(nn.Linear(16, 64), std=0.01),
            nn.ReLU(),
            # layer_init(nn.Linear(32, 32), std=0.01),
            # nn.ReLU(),
        )
        self.linear = nn.Sequential(
            layer_init(nn.Linear(64 + 512, 512), std=0.01),
            nn.ReLU(),
        )
        # self.find = nn.Sequential(
        #     layer_init(nn.Linear(512, 1), std=0.01),
        # )
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

    def get_action_and_value(self, x, detect, action=None):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1)
        action_mask = torch.zeros(detect.shape[0], env.action_space.n).to(device)
        action_mask[:, 0] = 1
        action_mask[:, 9] = 1
        detect = detect.squeeze(1)
        for i in range(detect.shape[0]):
            if detect[i, -1] < 0.5:
                action_mask[i, 10] = 1
            else:
                if detect[i, 11] < 0.65:
                    action_mask[i, 7] = 1  # no turn right
                if detect[i, 11] > 0.35:
                    action_mask[i, 8] = 1  # no turn left

        hidden = self.network(x / 255.0)
        # find = self.find(hidden)
        detect = self.detect(detect)

        hidden = torch.cat((hidden, detect), dim=1)
        hidden = self.linear(hidden)
        # hidden, self.rnn_hidden = self.rnn(hidden, self.rnn_hidden)

        logits = self.actor(hidden)

        logits = logits - action_mask * 1e8
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

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
    parser.add_argument("--num-steps", type=int, default=128 * 4,
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
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_name = f"{args.exp_name}__{args.env_id}__{args.seed}__{now_time}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    seed_it(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    env = EasyEnv(image_shape=(84, 168, 1))
    agent = Agent().to(device)
    loss_fd = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    load_model = False
    model_load_path = 'outputs/ppo_dict_model.pt'
    if load_model and os.path.exists(model_load_path):
        checkpoint = torch.load(model_load_path)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.network.load_state_dict(checkpoint['network'])
        agent.detect.load_state_dict(checkpoint['detect'])
        agent.linear.load_state_dict(checkpoint['linear'])
        agent.rnn.load_state_dict(checkpoint['rnn'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Model loaded from {}'.format(model_load_path))

    load_pretain_model = False
    pretrain_model_load_path = 'outputs/pretrain_model.pth'
    if load_pretain_model and os.path.exists(pretrain_model_load_path):
        checkpoint = torch.load(pretrain_model_load_path)
        agent.actor.load_state_dict(checkpoint['actor'])
        # agent.critic.load_state_dict(checkpoint['critic'])
        agent.network.load_state_dict(checkpoint['network'])
        # agent.optimizer.load_state_dict(checkpoint['optimizer'])
        print('pretrain Model loaded from {}'.format(pretrain_model_load_path))

    data_load_path = 'outputs/data_save/data.pt'

    data = torch.load(data_load_path)
    demo_obs = data['obs']
    demo_actions = data['actions']
    demo_detects = data['detects']
    print('data loaded from {}'.format(data_load_path))
    demo_batch_size = demo_obs.shape[0]
    demo_mini_batch_size = int(demo_batch_size // 4)

    obs = torch.zeros((args.num_steps, args.num_envs) + env.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + env.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    detects = torch.zeros((args.num_steps, args.num_envs) + (1, 16)).to(device)

    global_step = 0
    nepisode = 0
    best_reward = -np.inf
    rewards_episode = []
    time_start = time.time()
    next_obs = torch.Tensor(env.reset()).unsqueeze(0).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    next_detect = env.detect_info

    num_updates = args.total_timesteps // args.batch_size

    model_save_path = 'outputs/ppo_load_data_model.pt'

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = max(1.0 - (update - 1.0) / num_updates, 0.3)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            detects[step] = next_detect

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_detect)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            # if reward:
            #     print("reward", reward)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            done = np.expand_dims(done, 0)
            next_obs, next_done = torch.Tensor(next_obs).unsqueeze(0).to(device), torch.Tensor(done).to(device)
            next_detect = torch.Tensor(env.detect_info).unsqueeze(0).to(device)
            if info.get('game over', False):
                nepisode += 1
                reward_, length = env.reward, env.game_cnt
                time_end = time.time()
                time_cost = time_end - time_start
                print(
                    'Time steps so far: {}, episode so far: {}, '
                    'episode reward: {:.4f}, episode length: {}, '
                    'time cost: {:.2f} s'.format(global_step, nepisode, reward_, length, time_cost)
                )
                writer.add_scalar("charts/episodic_return", reward_, global_step)
                writer.add_scalar("charts/episodic_length", length, global_step)
                next_obs = torch.Tensor(env.reset()).unsqueeze(0).to(device)
                next_detect = torch.Tensor(env.detect_info).unsqueeze(0).to(device)
                rewards_episode.append(reward)
                if (nepisode + 1) % 10 == 0 and np.mean(rewards_episode[-15:]) >= best_reward - 0.5:
                    best_reward = np.mean(rewards_episode[-15:])
                    save_state = {
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'network': agent.network.state_dict(),
                        'detect': agent.detect.state_dict(),
                        'linear': agent.linear.state_dict(),
                        # 'find': agent.find.state_dict(),
                        # 'actor_detect': agent.model.actor_detect.state_dict(),
                        # 'critic_detect': ppo.model.critic_detect.state_dict(),
                        # 'linear_cat': ppo.model.linear_cat.state_dict(),
                        # "rnn": agent.rnn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(save_state, model_save_path)
                    print('model saved')

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_detect).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_detects = detects.reshape((-1,) + (1, 16))
        b_gt_find = b_detects[:, :, -1]
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        demo_inds = np.arange(demo_batch_size)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            np.random.shuffle(demo_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                # agent.reset_rnn_hidden()
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], [b_detects[s] for s in mb_inds], b_actions.long()[mb_inds])
                # logratio = newlogprob - b_logprobs[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                                       b_detects[mb_inds],
                                                                                       b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                # mb_gt_find = b_gt_find[mb_inds]

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Find loss
                # a = b_find - b_gt_find
                # fd_loss1 = torch.mean(torch.pow(a, 2))
                # fd_loss = loss_fd(newfind, mb_gt_find)

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef# + fd_loss * 0.5
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            for start in range(0, demo_batch_size, demo_mini_batch_size):
                end = start + demo_mini_batch_size
                mb_inds = demo_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(demo_obs[mb_inds], demo_detects[mb_inds],
                                                                              demo_actions.long()[mb_inds])
                demo_loss1 = -newlogprob
                demo_loss2 = -torch.clamp(newlogprob, -2.5, -0.45)
                demo_loss = 0.98**update * torch.max(demo_loss1, demo_loss2).mean()

                optimizer.zero_grad()
                demo_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # writer.add_scalar("losses/find_loss", fd_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - time_start)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - time_start)), global_step)
    env.close()
    writer.close()
