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
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized

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


class EasyEnv(TPSEnv):
    def __init__(self, ip_address="127.0.0.1", image_shape=(256, 512, 1), record=False,
                 yolo_weight='weights/best.pt', yolo_device='0'):
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
            "stage": np.zeros(1),
            "game over": np.zeros(1),
            "detect enemy": torch.zeros([1, 6]),
            "detect gun": torch.zeros([1, 6]),
            "detect exit": torch.zeros([1, 6]),
        }
        self.prev_state = self.state.copy()
        self.game_cnt = 0
        self.stay_cnt = 0
        self.max_step = 200  # per stage
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

        self.device = select_device(yolo_device)
        self.yolo_weight = yolo_weight
        self.yolo_model = attempt_load(self.yolo_weight, map_location=self.device)  # load FP32 yolo_model
        self.stride = int(self.yolo_model.stride.max())  # yolo_model stride
        self.imgsz = check_img_size(960, s=self.stride)  # check img_size
        self.augment = False

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.yolo_model.half()  # to FP16
        if self.device.type != 'cpu':
            self.yolo_model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.yolo_model.parameters())))  # run once

    def _get_obs(self):
        self.prev_state = self.state.copy()
        self.msg.msg_type = MSGType.QUERY_IMG
        reply = self.req.make_request(self.msg)
        img = msg_decode_as_image_bgr(reply)  # bgr
        img = (255 * (img / 255) ** 0.4).astype(np.uint8)  # gamma correction
        detect_info = self.detect(img)
        enemy, gun, exit_ = self.transform_detect(detect_info)
        self.state["detect enemy"] = enemy
        self.state["detect gun"] = gun
        self.state["detect exit"] = exit_
        # print(detect_info)
        if self.record:
            self.recorder(img)
        obs = self.transform_obs(img)
        self.msg.msg_type = MSGType.QUERY_STATUS
        reply = self.req.make_request(self.msg)
        status = msg_decode_as_float_vector(reply)
        self.state["has gun"] = np.array(status[0])  # 1 mean has gun, bool
        self.state["target"] = np.array(status[1])  # all targets number
        self.state["target hit"] = np.array(status[2])  # hit targets number
        # self.state["shoot nums"] = None  # shoot nums
        # self.state["shoot max"] = status[3]  # max shoot nums

        self.msg.msg_type = MSGType.QUERY_POS
        reply = self.req.make_request(self.msg)
        pos = msg_decode_as_float_vector(reply)
        # self.state["prev_position"] = self.state["position"]
        self.state["position"] = pos
        self.detect_info = np.zeros((1, 29))
        i = 0
        for key, value in self.state.items():
            if key != "detect enemy" and key != "detect gun" and key != "detect exit":
                self.detect_info[0, i:i + value.size] = value
                i += value.size
            else:
                self.detect_info[0, i:i + 6] = value
                i += 6
        self.detect_info = torch.from_numpy(self.detect_info).float().to(self.device)
        return obs

    def transform_obs(self, img):
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obs = obs.reshape(self.image_shape[0], self.image_shape[1], 1)
        # cv2.imshow("obs", obs)
        # cv2.waitKey(1)
        return obs

    def transform_detect(self, detect_info):
        enemy, gun, exit_ = torch.zeros([1, 6]), torch.zeros([1, 6]), torch.zeros([1, 6])
        if detect_info is not None:
            for i in range(detect_info.shape[0]):
                if detect_info[i, 0] == 2 and detect_info[i, 3] * detect_info[i, 4] > enemy[0, 3] * enemy[0, 4]:
                    a = detect_info[i, 3] * detect_info[i, 4]
                    b = enemy[0, 3] * enemy[0, 4]
                    enemy = detect_info[i, :].unsqueeze(0)
                elif detect_info[i, 0] == 1 and detect_info[i, 3] * detect_info[i, 4] > gun[0, 3] * gun[0, 4]:
                    gun = detect_info[i, :].unsqueeze(0)
                elif detect_info[i, 0] == 3 and detect_info[i, 3] * detect_info[i, 4] > exit_[0, 3] * exit_[0, 4]:
                    exit_ = detect_info[i, :].unsqueeze(0)
        return enemy, gun, exit_

    def detect(self, img0):
        # img0 = cv2.imread(path)  # BGR
        # cv2.imshow("img", img0)
        # cv2.waitKey(1)
        img = letterbox(img0, new_shape=self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        lines = torch.zeros((10, 6))  # output lines (tensor)
        # detect_matrix = torch.zeros(3, self.image_shape[0], self.image_shape[1])

        t0 = time.time()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.yolo_model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred)
        t3 = time_synchronized()
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # Process detections
        det = pred[0]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            i = 0
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # print(xywh)
                line = (cls + 1, *xywh, conf)
                lines[i] = torch.tensor(line).clone().detach()
                i += 1
        return lines

    def recorder(self, image):
        # cv2.imwrite(self.record_path + 'image_' + str(self.start_times) + ".png", image)
        cv2.imwrite(self.record_path + self.now_time + '_' + str(self.start_times) + ".jpg", image)
        self.start_times += 1

    def __del__(self):
        # self.reset()
        pass

    def _compute_reward(self):
        done = 0
        beta = 0.1
        gun_xywh = None
        enemy_xywh = None
        gun_reward = 1.5 * self.state["has gun"]
        reward = 0.044
        enemy_reward = 1.0 * self.state["target hit"]
        conf = 0
        pos_dist_reward = 0
        enemy_dist_reward = 0
        stage = 0  # the stage of the game, 0: no gun, 1: has gun, 2: shoot one enemy, 3: get new position,
        # 4: shoot all enemy, 5: exit
        kill_somebody = self.state["target hit"] - self.prev_state["target hit"]
        # if kill_somebody > 0:
        #     self.detect_enemy = None
        #     # print("kill somebody")

        if self.state["has gun"]:
            stage = 1
            # done = 1
            # print("stage 1")
            if self.state["target hit"] == 1:
                # stage = 2
                done = 1
                # print("stage 2")
                if self.state["position"][1] > 7300:
                    stage = 3
                    # done = 1
                    # print("stage 3")
            if self.state["target hit"] == self.state["target"]:
                stage = 4
                # done = 1
                # print("stage 4")
                if self.state["position"][0] < -2400:
                    stage = 5
                    done = 1
                    # print("stage 5")
        self.state["stage"][0] = stage if stage > self.state["stage"][0] else self.state["stage"][0]
        if self.state["detect enemy"][0][0] == 2 and self.prev_state["detect enemy"][0][0] == 2 and self.state[
            "has gun"] == 1:
            enemy_xywh = self.state["detect enemy"][0][1:5].numpy()
            prev_enemy_xywh = self.prev_state["detect enemy"][0][1:5].numpy()
            enemy_dist = np.linalg.norm((0.5, 0.5) - np.array([enemy_xywh[0], enemy_xywh[1]]))
            prev_enemy_dist = np.linalg.norm((0.5, 0.5) - np.array([prev_enemy_xywh[0], prev_enemy_xywh[1]]))
            enemy_dist_reward = 0.1 * ((prev_enemy_dist - enemy_dist) / 0.5) * 2
        if self.state["stage"][0] == 0:
            gun_pt = [852, 4200.0]  # where the gun is
            dist = np.linalg.norm(self.state["position"][0:2] - gun_pt)  # distance
            threshold_dist = 1500  # reset if distance is too far
            reward_dist = 700  # reward grows faster when distance is smaller
            if dist > threshold_dist:
                done = 1
                pos_dist_reward = -1
            else:
                pos_dist_reward += -beta * math.log((dist / reward_dist), 1.5)
        elif self.state["stage"][0] == 1:
            # dist = abs(self.state['position'][1] - 5000)/1300
            # pos_dist_reward = (1 - dist) * 3
            pos_dist_reward = 0
        elif self.state["stage"][0] == 2:
            enemy_pt = [800, 8000]
            dist = np.linalg.norm(self.state["position"][0:2] - enemy_pt)
            pos_dist_reward = -beta * math.log((dist / 4000), 1.5)
        elif self.state["stage"][0] == 3:
            # dist = abs(self.state['position'][1] - 8000) / 3800
            enemy_pt = [-200, 8000]
            exit_pt = [-2630, 7911]
            dist = np.linalg.norm(self.state["position"][0:2] - enemy_pt)
            angle = math.atan2(self.state["position"][1] - exit_pt[1],
                               self.state["position"][0] - exit_pt[0]) * 180 / math.pi
            target_angle = angle - 180 if angle > 0 else angle + 180
            delta_angle = abs(self.state["position"][4] - target_angle) if target_angle * self.state["position"][
                4] > 0 else 360 - abs(self.state["position"][4] - target_angle)
            if delta_angle > 180:
                delta_angle = 360 - delta_angle
            if dist > 3000:
                done = 1
                pos_dist_reward = -1
            else:
                pos_dist_reward = 0.5 * (1 - (dist / 2000)) + 0.3 + 0.2 * (0.5 - (delta_angle / 180))
                # print("pos_dist_reward: ", (0.5 - (delta_angle / 180)))
        elif self.state["stage"][0] == 4:
            exit_pt = [-2630, 7911]
            dist = np.linalg.norm(self.state["position"][0:2] - exit_pt)  # distance
            pos_dist_reward = -beta * math.log((dist / 3800), 1.5) + 0.6
        elif self.state["stage"][0] == 5:
            pos_dist_reward = 2.3
        if self.game_cnt > (self.max_step * (stage + 1)) or self.stay_cnt > 100 * (stage + 1):
            done = 1
            # reward -= 5

        if done == 1:
            self.state["game over"] = np.array([1])
        # print(dist)
        reward += gun_reward + enemy_reward + pos_dist_reward - self.stay_cnt * 0.001 - self.game_cnt * 0.001
        # print("reward: ", reward)
        reward_delta = reward - self.reward + enemy_dist_reward
        # if abs(reward_delta) < 1:
        #     reward_delta *= 10
        self.reward = reward
        return reward_delta, done

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
        self.state = {
            "position": np.zeros(5),
            # "prev_position": np.zeros(5),
            "has gun": np.zeros(1),
            "target hit": np.zeros(1),
            "target": np.zeros(1),
            # "shoot nums": None,
            "shoot max": np.zeros(1),
            # "collision": False,
            "stage": np.zeros(1),
            "game over": np.zeros(1),
            "detect enemy": torch.zeros([1, 6]),
            "detect gun": torch.zeros([1, 6]),
            "detect exit": torch.zeros([1, 6]),
        }
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
            time.sleep(0.005)
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
        elif action == 6:  # rotate left
            self.stay_cnt += 1
            self.msg.msg_content = cmd_rotate([-1, 0]).to_bytes()
            reply = self.req.make_request(self.msg)
        elif action == 7:  # jump, Maintain inertia, full action take about 0.8s
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
        elif action == 8:
            if self.state["has gun"]:
                self.msg.msg_content = cmd_event([0, 0, 1, 0]).to_bytes()
                reply = self.req.make_request(self.msg)
                self.rifle_cnt += 1
                time.sleep(0.3)
            else:
                self.msg.msg_content = cmd_move([0, 0]).to_bytes()
                reply = self.req.make_request(self.msg)
                self.stay_cnt += 1
                time.sleep(0.05)


if __name__ == "__main__":
    # a = torch.ones(32, 30, 6)
    # b = torch.nonzero(a[:, :, 0])
    # c = len(b)

    test = EasyEnv(record=False, image_shape=(128, 256, 1), yolo_weight='../weights/best.pt')
    test.reset()
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
        # e = torch.tensor(e, dtype=    torch.float)
        # if c:
        #     test.reset()
        # plt.imshow(img1, cmap='gray')
        # plt.show()
        time.sleep(0.5)
        if c:
            test.reset()
