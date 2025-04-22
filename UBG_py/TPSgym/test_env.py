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


class TestEnv(TPSEnv):
    def __init__(self, ip_address="127.0.0.1", image_shape=(256, 512, 1), record=False,
                 ):
        super().__init__(image_shape)
        self.image_shape = image_shape
        self.start_times = 0
        self.state = {
            "position": np.zeros(5),
            # "prev_position": np.zeros(5),
            "has gun": np.zeros(1),
            "target hit": np.zeros(1),
            "target": np.zeros(1),
            # "shoot nums": None,
            "shoot max": np.zeros(1),
            # "collision": False,
            "game over": np.zeros(1),
            "detect enemy": torch.zeros([1, 6]),
            "detect gun": torch.zeros([1, 6]),
            "detect exit": torch.zeros([1, 6]),
        }
        self.game_cnt = 0
        self.stay_cnt = 0
        self.max_step = 500
        self.rifle_cnt = 0
        self.max_rifle = 30
        self.reward = 0
        self.req = RequestSender(ip_address)
        # self.action_space = spaces.MultiDiscrete((7, 2))
        self.action_space = spaces.Discrete(9)
        self.msg = MSG()
        self.record = record
        self.now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.record_path = 'record/' + self.now_time + "/"
        if self.record:
            os.mkdir(self.record_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_obs(self):
        self.msg.msg_type = MSGType.QUERY_IMG
        reply = self.req.make_request(self.msg)
        img = msg_decode_as_image_bgr(reply)  # bgr
        img = (255 * (img / 255) ** 0.5).astype(np.uint8)  # gamma correction
        if self.record:
            self.recorder(img)
        obs = self.transform_obs(img)
        self.msg.msg_type = MSGType.QUERY_STATUS
        reply = self.req.make_request(self.msg)
        status = msg_decode_as_float_vector(reply)
        self.state["has gun"] = status[0]  # 1 mean has gun
        self.state["target"] = status[1]  # all targets number
        self.state["target hit"] = status[2]  # hit targets number
        # self.state["shoot nums"] = None  # shoot nums
        # self.state["shoot max"] = status[3]  # max shoot nums

        self.msg.msg_type = MSGType.QUERY_POS
        reply = self.req.make_request(self.msg)
        pos = msg_decode_as_float_vector(reply)
        # self.state["prev_position"] = self.state["position"]
        self.state["position"] = pos
        self.detect_info = []
        for key, value in self.state.items():
            self.detect_info.append(torch.tensor(self.state[key]).to(self.device))
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
        beta = 1
        gun_reward = 0
        reward = 0
        enemy_reward = 0

        gun_pt = [852, 4200.0]  # where the gun is
        dist = np.linalg.norm(self.state["position"][0:2] - gun_pt)  # distance
        threshold_dist = 1800  # reset if distance is too far
        reward_dist = 700  # reward grows faster when distance is smaller
        if not self.state["has gun"]:
            if dist > threshold_dist:
                done = 1
                gun_reward = -8
            else:
                gun_reward = -beta * math.log((dist / reward_dist), 1.5)
        else:
            done = 1
            gun_reward = 15

        enemy_reward = self.state["target hit"] * 10

        if self.state["target hit"] == 1:
            done = 1

        if self.game_cnt > self.max_step or self.stay_cnt > 100:
            done = 1
            reward -= 5
            # pass
        if done == 1:
            self.state["game over"] = np.ones(1)
        # print(dist)
        reward += gun_reward + enemy_reward - self.stay_cnt * 0.01 - self.rifle_cnt * 0.1 - self.game_cnt * 0.01

        reward_det = reward - self.reward
        # if abs(reward_det) < 1:
        #     reward_det = reward_det * 10
        self.reward = reward
        return reward_det, done

    def close(self):
        self.reset()

    def step(self, action):
        self._do_action(action)
        self.game_cnt += 1
        # obs, detect_info = self._get_obs()
        obs = self._get_obs()
        reward, done = self._compute_reward()
        print("action: ", action, "  reward: ", reward)
        # return obs, self.detect_info, reward, done, self.state
        return obs, reward, done, self.state

    def reset(self):
        self.game_cnt = 0
        self.stay_cnt = 0
        self.reward = 0
        self.rifle_cnt = 0
        self.state["game over"] = np.zeros(1)
        self.msg.msg_type = MSGType.SEND_CTRL_CMD
        self.msg.msg_content = cmd_event([1, 0, 0, 0]).to_bytes()
        reply = self.req.make_request(self.msg)
        self._do_action(0)
        time.sleep(0.5)
        return self._get_obs()

    def _do_action(self, action):
        self.msg.msg_type = MSGType.SEND_CTRL_CMD
        if action == 0:  # stay
            self.stay_cnt += 1
            self.msg.msg_content = cmd_move([0, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action == 1:  # move forward
            self.msg.msg_content = cmd_move([1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action == 2:  # move backward
            self.msg.msg_content = cmd_move([-1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action == 3:  # move right
            self.msg.msg_content = cmd_move([0, 1]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action == 4:  # move left
            self.msg.msg_content = cmd_move([0, -1]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action == 5:  # rotate right
            self.stay_cnt += 1
            self.msg.msg_content = cmd_rotate([1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            # reply = self.req.make_request(self.msg)
        elif action == 6:  # rotate left
            self.stay_cnt += 1
            self.msg.msg_content = cmd_rotate([-1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            # reply = self.req.make_request(self.msg)
        elif action == 7:  # jump, Maintain inertia, full action take about 0.8s
            self.stay_cnt += 1
            temp_action = self.msg.msg_content
            self.msg.msg_content = cmd_event([0, 1, 0, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.6)
            if temp_action in [cmd_move([1, 0]).to_bytes(), cmd_move([-1, 0]).to_bytes(),
                               cmd_move([0, 1]).to_bytes(), cmd_move([0, -1]).to_bytes()]:
                self.msg.msg_content = temp_action
                reply = self.req.make_request(self.msg)
                time.sleep(0.01)
        elif action == 8:
            if self.state["has gun"]:
                self.msg.msg_content = cmd_event([0, 0, 1, 0]).to_bytes()
                reply = self.req.make_request(self.msg)
                self.rifle_cnt += 1
                time.sleep(0.5)
            else:
                self.msg.msg_content = cmd_move([0, 0]).to_bytes()
                reply = self.req.make_request(self.msg)
                self.stay_cnt += 1
                time.sleep(0.05)


if __name__ == "__main__":
    # a = torch.ones(32, 30, 6)
    # b = torch.nonzero(a[:, :, 0])
    # c = len(b)

    test = TestEnv(record=False, image_shape=(128, 256, 1))
    # while True:
    #     # aaa = test.step(1)
    #     # n = test.state["position"]
    #     # print("1", n)
    #     aaa, b, c, d = test.step(0)
    #     n = test.state["position"]
    #     print("2", n)
    #     time.sleep(2)
    while True:
        img1, b, c, d = test.step(0)
        # a = torch.tensor(img1, dtype=torch.float)
        # a = a.unsqueeze(0)
        # g = e[:, 5]
        # f = torch.nonzero(e[:, 5]).max().item() + 1
        # h = e[:f]
        # e = torch.tensor(e, dtype=torch.float)
        # if c:
        #     test.reset()
        # plt.imshow(img1, cmap='gray')
        # plt.show()
        time.sleep(1)
        if c:
            test.reset()
            time.sleep(1)
