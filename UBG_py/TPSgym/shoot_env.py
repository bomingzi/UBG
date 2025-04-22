from jycomm import RequestSender, MSG, MSGType
from jycomm import msg_decode_as_float_vector
from jycomm import msg_decode_as_string
from jycomm import msg_decode_as_image_bgr
from TPSgym.control_cmd import CTRLCMD, cmd_event, cmd_move, cmd_rotate

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


class ShootEnv(TPSEnv):
    def __init__(self, ip_address="127.0.0.1", image_shape=(84, 84, 1), record=False):
        super().__init__(image_shape)
        self.image_shape = image_shape
        self.start_times = 0
        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "has gun": False,
            "target hit": np.zeros(1),
            "target": np.zeros(1),
            "shoot nums": None,
            "shoot max": np.zeros(1),
            "collision": False,
        }
        self.game_cnt = 0
        self.stay_cnt = 0
        self.rifle_cnt = 0
        self.max_rifle = 30
        self.max_step = 150
        self.req = RequestSender(ip_address)
        self.action_space = spaces.MultiDiscrete((11, 2))
        self.msg = MSG()
        self.record = record
        self.now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.record_path = 'record/' + self.now_time + "/"
        if self.record:
            os.mkdir(self.record_path)
        # cv2.namedWindow("obs")

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
        self.state["has gun"] = (status[0] == 1)  # 1 means has gun, bool
        self.state["target"] = status[1]  # all targets number
        self.state["target hit"] = status[2]  # hit targets number
        # self.state["shoot nums"] = None  # shoot nums
        # self.state["shoot max"] = status[3]  # max shoot nums

        self.msg.msg_type = MSGType.QUERY_POS
        reply = self.req.make_request(self.msg)
        pos = msg_decode_as_float_vector(reply)
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = pos

        return obs

    def transform_obs(self, img):
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obs = obs.reshape(self.image_shape[0], self.image_shape[1], 1)
        # cv2.imshow("obs", obs)
        # cv2.waitKey(1)
        return obs

    def recorder(self, image):
        cv2.imwrite(self.record_path + 'image_' + str(self.start_times) + ".png", image)
        self.start_times += 1

    def __del__(self):
        self.reset()

    def _compute_reward(self):
        detect_info = self.detect_info
        done = 0
        beta = 1
        gun_xywh = None
        gun_reward = 0
        conf = 0
        for line in detect_info:
            if line[5] < 0.5:
                break
            if line[0] == 2:
                gun_xywh = line[1:5].numpy()
                conf = line[5].numpy()
                break
        if self.game_cnt > self.max_step:
            done = 1
            reward = -8 + self.state["target hit"] * 4
        elif self.state["target hit"] == 2:
            done = 1
            reward = 10
        elif self.rifle_cnt > self.max_rifle:
            done = 1
            reward = -10 + self.state["target hit"] * 4
        else:
            done = 0
            reward = - self.game_cnt / 30 + self.state["target hit"] * 4 - self.rifle_cnt / 30
        return reward, done

    def close(self):
        pass

    def step(self, action):
        self._do_action(action)
        self.game_cnt += 1
        obs = self._get_obs()
        reward, done = self._compute_reward()
        print("action: ", action, "  reward: ", reward)

        return obs, reward, done, self.state

    def reset(self):
        self.game_cnt = 0
        self.stay_cnt = 0
        self.rifle_cnt = 0
        self.msg.msg_type = MSGType.SEND_CTRL_CMD
        self.msg.msg_content = cmd_event([1, 0, 0, 0]).to_bytes()
        reply = self.req.make_request(self.msg)
        self._do_action([0, 0])
        return self._get_obs()

    def _do_action(self, action):
        self.msg.msg_type = MSGType.SEND_CTRL_CMD
        if action[0] == 0:  # stay
            self.stay_cnt += 1
            self.msg.msg_content = cmd_move([0, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.005)
        elif action[0] == 1:  # move forward
            self.msg.msg_content = cmd_move([1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action[0] == 2:  # move backward
            self.msg.msg_content = cmd_move([-1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action[0] == 3:  # move right
            self.msg.msg_content = cmd_move([0, 1]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action[0] == 4:  # move left
            self.msg.msg_content = cmd_move([0, -1]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.05)
        elif action[0] == 5:  # rotate right
            self.stay_cnt += 1
            self.msg.msg_content = cmd_rotate([1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            reply = self.req.make_request(self.msg)
            reply = self.req.make_request(self.msg)
            reply = self.req.make_request(self.msg)
        elif action[0] == 6:  # rotate left
            self.stay_cnt += 1
            self.msg.msg_content = cmd_rotate([-1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            reply = self.req.make_request(self.msg)
            reply = self.req.make_request(self.msg)
            reply = self.req.make_request(self.msg)
        elif action[0] == 8:  # jump, Maintain inertia, full action take about 0.8s
            self.stay_cnt += 1
            temp_action = self.msg.msg_content
            self.msg.msg_content = cmd_event([0, 1, 0, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            time.sleep(0.8)
            if temp_action in [cmd_move([1, 0]).to_bytes(), cmd_move([-1, 0]).to_bytes(),
                               cmd_move([0, 1]).to_bytes(), cmd_move([0, -1]).to_bytes()]:
                self.msg.msg_content = temp_action
                reply = self.req.make_request(self.msg)
                time.sleep(0.01)
        elif action[0] == 7:
            self.msg.msg_content = cmd_event([0, 0, 1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
            self.rifle_cnt += 1


if __name__ == "__main__":
    env = ShootEnv()
    obs = env.reset()
    action_space = spaces.MultiDiscrete([11, 2])
    while True:
        for i in range(1000):
            action = action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()
