from jycomm import RequestSender, MSG, MSGType
from jycomm import msg_decode_as_float_vector
from jycomm import msg_decode_as_string
from jycomm import msg_decode_as_image_bgr
from TPSgym.control_cmd import CTRLCMD, cmd_event, cmd_move, cmd_rotate

import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

import numpy as np
import math
import time
import datetime

import gym
from gym import spaces
from TPSgym.TPS_env import TPSEnv
import matplotlib.pyplot as plt
import cv2
import os


class UBGEnv(TPSEnv):
    def __init__(self, ip_address="127.0.0.1", image_shape=(256, 512, 1), record=False,
                 port=14514):
        super().__init__(image_shape)
        self.obs = None
        self.img_depth = None
        self.image_shape = image_shape
        self.start_times = 0
        self.state = {
            "position": np.zeros(6),
            # "prev_position": np.zeros(6),
            "has gun": np.zeros(1),
            "total ammo": np.zeros(1),  # total ammo
            "ammo": np.zeros(1),  # ammo
            "helmet": np.zeros(1),  # helmet
            "vest": np.zeros(1),  # vest
            "HP": np.array([100]),  # HP
            "total damage": np.zeros(1),  # total damage
            "alive enemy num": np.array([3]),

            "stage": np.zeros(1),
            "game over": np.zeros(1),

        }
        self.prev_state = self.state.copy()
        self.game_cnt = 0
        self.stay_cnt = 0
        self.max_step = 200  # per stage
        self.req_ipaddr = ip_address
        self.req_port = port
        self.reward = 0
        # self.msg = MSG()
        self.req = RequestSender(ip_address)
        # self.action_space = spaces.MultiDiscrete((9, 11, 2, 2, 5))
        self.action_space = spaces.MultiDiscrete((5, 11, 2))
        # self.action_space = spaces.Discrete(11)
        # self.msg = MSG()
        self.record = record
        self.now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.record_path = 'record/' + self.now_time + "/"
        if self.record:
            os.mkdir(self.record_path)

    def reset_connection(self):
        if self.req is not None:
            del self.req
            self.req = RequestSender(self.req_ipaddr, self.req_port)

    def test_ue(self) -> bool:
        msg = MSG()
        msg.msg_type = MSGType.EXTEND_TEST
        rep = self.req.make_request(msg)
        if rep is None:
            return False
        if rep.msg_type == MSGType.RESPONSE:
            return True
        return False

    def _get_obs(self):
        self.prev_state = self.state.copy()
        msg = MSG()
        msg.msg_type = MSGType.EXTEND_1
        reply = self.req.make_request(msg)
        if reply is None:
            return None
        img = msg_decode_as_image_bgr(reply)  # bgr
        msg = MSG()
        msg.msg_type = MSGType.QUERY_IMG
        reply = self.req.make_request(msg)
        if reply is None:
            return None
        self.img_depth = msg_decode_as_image_bgr(reply)  # depth

        if self.record:
            self.recorder(img)
        obs = self.transform_obs(img)
        msg = MSG()
        msg.msg_type = MSGType.QUERY_STATUS
        reply = self.req.make_request(msg)
        if reply is None:
            return None
        self.enemy_status = msg_decode_as_float_vector(reply)
        self.state["alive enemy num"] = np.array(
            np.sum(np.array([self.enemy_status[4], self.enemy_status[10], self.enemy_status[16]]) > 0),
            dtype=np.float32, ndmin=1)
        msg = MSG()
        msg.msg_type = MSGType.QUERY_POS
        reply = self.req.make_request(msg)
        if reply is None:
            return None
        status = msg_decode_as_float_vector(reply)
        self.state["position"] = np.array(status[0:6], dtype=np.float32, ndmin=1)
        if self.state["position"][5] < 0:
            self.state["position"][5] += 360
        self.state["has gun"] = np.array(status[6], ndmin=1)  # 1 mean has gun
        self.state["total ammo"] = np.array(status[7], ndmin=1)  # total ammo
        self.state["ammo"] = np.array(status[8], ndmin=1)  # ammo
        self.state["helmet"] = np.array(status[9], ndmin=1)  # helmet
        self.state["vest"] = np.array(status[10], ndmin=1)  # vest
        self.state["HP"] = np.array(status[11], ndmin=1)  # HP
        self.state["total damage"] = np.array(status[12], ndmin=1)  # total damage
        return obs

    def transform_obs(self, img):
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obs = obs.reshape(self.image_shape[0], self.image_shape[1], 1)
        # cv2.imshow("obs", obs)
        # cv2.waitKey(1)

        return obs

    def recorder(self, image):
        # cv2.imwrite(self.record_path + 'image_' + str(self.start_times) + ".png", image)
        cv2.imwrite(self.record_path + self.now_time + '_' + str(self.start_times) + ".jpg", image)
        self.start_times += 1

    def __del__(self):
        # self.reset()
        pass

    def _compute_reward(self):
        done = 0

        gun_reward = 1.5 * self.state["has gun"]
        reward = 0
        enemy_reward = 5.0 * (5 - self.state["alive enemy num"])
        damage_reward = 0.02 * self.state["total damage"]
        vest_reward = 1 * self.state["vest"]
        helmet_reward = 0.5 * self.state["helmet"]
        ammo_reward = 0.01 * self.state["ammo"] + 0.005 * self.state["total ammo"]
        HP_reward = 0.01 * (self.state["HP"] - 100)

        stage = 0  # the stage of the game, 0: no gun, 1: has gun, 2: shoot one enemy, 3: get new position,
        if self.state["has gun"]:
            stage = 1
        self.state["stage"][0] = stage if stage > self.state["stage"][0] else self.state["stage"][0]

        if self.game_cnt > (
                self.max_step * (stage + 1 + (3 - self.state["alive enemy num"]))) or self.stay_cnt > 100 * (
                stage + 1 + (3 - self.state["alive enemy num"])):
            done = 1
            # reward -= 5
        if self.state["HP"] < 1 or self.state["alive enemy num"] == 0:
            done = 1

        if done == 1:
            self.state["game over"] = np.array([1])

        reward += gun_reward + enemy_reward + helmet_reward + vest_reward \
                  + HP_reward + damage_reward + ammo_reward - self.stay_cnt * 0.002 - self.game_cnt * 0.001

        reward = reward.__float__()

        reward_delta = reward - self.reward

        self.reward = reward
        return reward_delta, done

    def close(self):
        self.reset()

    def step(self, action):
        action_done = self._do_action(action)
        # if not action_done:
            # return None
        self.game_cnt += 1
        # obs, detect_info = self._get_obs()
        self.obs = self._get_obs()
        if self.obs is None:
            return None
        reward, done = self._compute_reward()
        print("action: ", action, "  reward: ", reward)
        # return obs, self.detect_info, reward, done, self.state
        return self.obs, reward, done, self.state

    def reset(self):
        self.game_cnt = 0
        self.stay_cnt = 0
        self.reward = 0

        self.state = {
            "position": np.zeros(6),
            # "prev_position": np.zeros(6),
            "has gun": np.zeros(1),
            "total ammo": np.zeros(1),  # total ammo
            "ammo": np.zeros(1),  # ammo
            "helmet": np.zeros(1),  # helmet
            "vest": np.zeros(1),  # vest
            "HP": np.array([100]),  # HP
            "total damage": np.zeros(1),  # total damage
            "alive enemy num": np.array([3]),

            "stage": np.zeros(1),
            "game over": np.zeros(1),
            "detect enemy": torch.zeros([1, 6]),
        }
        msg = MSG()
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_move([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).to_bytes()
        reply = self.req.make_request(msg)

        time.sleep(5.0)
        action_done = self._do_action(np.zeros(3))
        if not action_done:
            return None
        self.obs = self._get_obs()
        if self.obs is None:
            return None
        return self.obs

    def _do_action(self, action) -> bool:
        action_cmd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # [0:move forward/backward, 1:move right/left, 2:yaw, 3:pitch, 4:crouch蹲,
        # 5:prone趴, 6:jump, 7:fire, 8:yaw_angle, 9:reload, 10:health, 11:restart]
        msg = MSG()
        msg.msg_type = MSGType.SEND_CTRL_CMD

        if action[1] == 1:  # rotate right lightly
            # self.stay_cnt += 1
            action_cmd[8] = 1
        elif action[1] == 2:  # rotate left lightly
            # self.stay_cnt += 1
            action_cmd[8] = -1
        elif action[1] == 3:  # rotate right
            # self.stay_cnt += 1
            action_cmd[8] = 2
        elif action[1] == 4:  # rotate left
            # self.stay_cnt += 1
            action_cmd[8] = -2
        elif action[1] == 5:  # rotate right
            self.stay_cnt += 0.2
            action_cmd[8] = 4
        elif action[1] == 6:  # rotate left
            self.stay_cnt += 0.2
            action_cmd[8] = -4
        elif action[1] == 7:  # rotate right
            self.stay_cnt += 1
            action_cmd[8] = 10
        elif action[1] == 8:  # rotate left
            self.stay_cnt += 1
            action_cmd[8] = -10
        elif action[1] == 9:  # rotate right
            self.stay_cnt += 3
            action_cmd[8] = 30
        elif action[1] == 10:  # rotate left
            self.stay_cnt += 3
            action_cmd[8] = -30
        if action[2] == 1:
            if self.state["has gun"] and self.state["ammo"] > 0:
                action_cmd[7] = 1
            else:
                pass

        if action[0] == 0:  # stay
            self.stay_cnt += 1
        elif action[0] == 1:  # move forward
            action_cmd[0] = 1
        elif action[0] == 2:  # move backward
            action_cmd[0] = -1
        elif action[0] == 3:  # move right
            action_cmd[1] = 1
            time.sleep(0.05)
        elif action[0] == 4:  # move left
            action_cmd[1] = -1
        msg.msg_content = cmd_move(action_cmd).to_bytes()
        reply = self.req.make_request(msg)
        time.sleep(0.05)

        if reply is None:
            return False

        return True


if __name__ == "__main__":
    game = UBGEnv(record=False, image_shape=(256, 512, 1))
    action_space = game.action_space
    # action space, MultiDiscrete((9, 11, 2, 2, 5)), move, rotate, fire, jump, skill

    for i in range(10):  # loop over 10 episodes.
        # reset the game and get the screen buffer and game variables
        obs = game.reset()
        depth = game.img_depth
        misc = game.state
        while not misc["game over"][0]:
            # perform a random action, here we just sample one from the action space.
            action = action_space.sample()

            next_obs, reward, done, info = game.step(action)
            next_depth = game.img_depth
            next_misc = game.state

            # do something with the observation and reward...
        print("reward: ", game.reward)
