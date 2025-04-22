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
from utils.plots import plot_one_box

class MaEnv(TPSEnv):
    def __init__(self, ip_address="127.0.0.1", image_shape=(256, 512, 1), record=False,
                 yolo_weight='weights/best.pt', yolo_device='0', port=14514):
        super().__init__(image_shape)
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
            "detect enemy": torch.zeros([1, 6]),
            "guide": np.zeros(1),
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
        self.action_space = spaces.MultiDiscrete((5, 11, 2))
        # self.action_space = spaces.Discrete(11)
        #self.msg = MSG()
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
        img_depth = msg_decode_as_image_bgr(reply)  # bgr
        # cv2.imshow("dep", img_depth)
        # cv2.waitKey(1)
        # img = (255 * (img / 255) ** 0.4).astype(np.uint8)  # gamma correction
        detect_info, detect_matrix = self.detect(img, img_depth)
        enemy = self.transform_detect(detect_info)
        self.state["detect enemy"] = enemy
        # self.state["detect gun"] = gun
        # self.state["detect exit"] = exit_
        # print(detect_info)
        if self.record:
            self.recorder(img)
        obs, self.detect_map = self.transform_obs(img, detect_matrix)
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
        self.state["guide"] = np.array(status[13], ndmin=1)
        print("guide ", self.state["guide"])
        self.game_info = np.zeros((1, 22))
        i = 0
        for key, value in self.state.items():
            if key != "detect enemy":
                self.game_info[0, i:i + value.size] = value
                i += value.size
            else:
                self.game_info[0, i:i + 6] = value
                i += 6
        self.game_info = torch.from_numpy(self.game_info).float().to(self.device)
        self.detect_map = torch.from_numpy(self.detect_map).float().to(self.device)
        return obs

    def transform_obs(self, img, detect_matrix):
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        obs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obs = obs.reshape(self.image_shape[0], self.image_shape[1], 1)
        # cv2.imshow("obs", obs)
        # cv2.waitKey(1)
        detect_map = cv2.resize(detect_matrix, (self.image_shape[1], self.image_shape[0]))
        # cv2.imshow("detect_map", detect_map[:, :, 0] / 255)
        # cv2.waitKey(1)
        return obs, detect_map

    def transform_detect(self, detect_info):
        enemy = torch.zeros([1, 6])
        if detect_info is not None:
            for i in range(detect_info.shape[0]):
                if detect_info[i, 0] == 6 and detect_info[i, 3] * detect_info[i, 4] > enemy[0, 3] * enemy[0, 4]:
                    enemy = detect_info[i, :].unsqueeze(0)
        return enemy

    def detect(self, img0, img_depth):
        # img0 = cv2.imread(path)  # BGR
        # cv2.imshow("img", img0)
        # cv2.waitKey(1)
        img = letterbox(img0, new_shape=self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        lines = torch.zeros((10, 6))  # output lines (tensor)
        detect_matrix = np.ones([img_depth.shape[0], img_depth.shape[1], 2]) * 255

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

        for i, det in enumerate(pred):  # detections per image
            for *xyxy, conf, cls in reversed(det):
                print(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                # if True:  # Add bbox to image
                #     label = 'aaa'
                #     plot_one_box(xyxy, img, color=(255, 0, 0), label=label, line_thickness=1)
        # cv2.imshow("aaa", img)
        # gn = torch.tensor(self.image_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # Process detections
        det = pred[0]
        # gun ammo health helmet vest enemy
        if len(det):
            # Rescale boxes from img_size to image_shape size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.image_shape[0:2]).round()
            i = 0
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if cls == 5:  # enemy
                    detect_matrix[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), 0] = img_depth[
                                                                                             int(xyxy[1]):int(xyxy[3]),
                                                                                             int(xyxy[0]):int(xyxy[2]),
                                                                                             0]
                else:  # gun ammo health helmet vest
                    detect_matrix[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), 1] = img_depth[
                                                                                             int(xyxy[1]):int(xyxy[3]),
                                                                                             int(xyxy[0]):int(xyxy[2]),
                                                                                             0]
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # print(xywh)
                line = (cls + 1, *xywh, conf)
                lines[i] = torch.tensor(line).clone().detach()
                i += 1
        # cv2.imshow("dep", detect_matrix[:, :, 0]/255)
        # cv2.waitKey(1)
        # detect_map = torch.from_numpy(detect_matrix).float().to(self.device)
        return lines, detect_matrix

    def recorder(self, image):
        # cv2.imwrite(self.record_path + 'image_' + str(self.start_times) + ".png", image)
        cv2.imwrite(self.record_path + self.now_time + '_' + str(self.start_times) + ".jpg", image)
        self.start_times += 1

    def __del__(self):
        # self.reset()
        pass

    def _compute_reward(self):
        done = 0
        beta = 0.1  # 0.1 normal
        # return 0, 1
        gun_xywh = None
        enemy_xywh = None

        gun_reward = 1.5 * self.state["has gun"]
        reward = 0.0075
        enemy_reward = 5.0 * (3 - self.state["alive enemy num"])
        damage_reward = 0.02 * self.state["total damage"]
        vest_reward = 1 * self.state["vest"]
        helmet_reward = 0.5 * self.state["helmet"]
        ammo_reward = 0.01 * self.state["ammo"] + 0.005 * self.state["total ammo"]
        HP_reward = 0.01 * (self.state["HP"] - 100)
        guide_reward = 0.01 * (self.state["guide"])
        pos_dist_reward = 0
        enemy_dist_reward = 0
        stage = 0  # the stage of the game, 0: no gun, 1: has gun, 2: shoot one enemy, 3: get new position,
        # 4: shoot all enemy, 5: exit

        # kill_somebody = self.prev_state["alive enemy num"] - self.state["alive enemy num"]
        pos_dist_reward_ = 0
        # if kill_somebody > 0:
        #     self.detect_enemy = None
        #     # print("kill somebody")
        gun_pt = [959, 4621.0]
        if self.state["has gun"]:
            stage = 1
        self.state["stage"][0] = stage if stage > self.state["stage"][0] else self.state["stage"][0]
        if self.state["detect enemy"][0][0] == 6 and self.prev_state["detect enemy"][0][0] == 6 and self.state[
            "has gun"] == 1:
            enemy_xywh = self.state["detect enemy"][0][1:5].numpy()
            prev_enemy_xywh = self.prev_state["detect enemy"][0][1:5].numpy()
            enemy_dist = np.linalg.norm((0.5, 0.5) - np.array([enemy_xywh[0], enemy_xywh[1]]))
            prev_enemy_dist = np.linalg.norm((0.5, 0.5) - np.array([prev_enemy_xywh[0], prev_enemy_xywh[1]]))
            enemy_dist_reward = beta * ((prev_enemy_dist - enemy_dist) / 0.5) * 2  # kaolv (1-x1)^2 - (1-x2)^2
        if self.state["stage"][0] == 0:
            gun_dist = np.linalg.norm(self.state["position"][0:2] - gun_pt)  # distance
            # prev_gun_dist = np.linalg.norm(self.prev_state["position"][0:2] - gun_pt)
            threshold_dist = 1800  # reset if distance is too far
            reward_dist = 700  # reward grows faster when distance is smaller
            if gun_dist > threshold_dist:
                print("gun done")
                done = 1
                pos_dist_reward = -1
            else:
                pos_dist_reward = - beta * math.log((gun_dist / reward_dist), 1.5)
        elif self.state["stage"][0] == 1:
            enemy_dist = np.array([self.enemy_status[0], self.enemy_status[6], self.enemy_status[12]])
            min_dist_idx = np.argmin(enemy_dist)
            min_dist = enemy_dist[min_dist_idx]
            min_pt = np.array([self.enemy_status[min_dist_idx * 6 + 1], self.enemy_status[min_dist_idx * 6 + 2]])

            angle = math.atan2(self.state["position"][1] - min_pt[1],
                               self.state["position"][0] - min_pt[0]) * 180 / math.pi
            target_angle = angle - 180 if angle > 0 else angle + 180
            target_angle = target_angle if target_angle > 0 else target_angle + 360

            delta_angle = abs(self.state["position"][5] - target_angle) if abs(
                self.state["position"][5] - target_angle) < 180 else 360 - abs(
                self.state["position"][5] - target_angle)
            # print("delta_angle: ", delta_angle)

            # print("min_dist: ", min_dist)
            if min_dist < 10000:
                pos_dist_reward = 0.5 * (0.5 - (delta_angle / 180)) + 0.5 * (1 - abs(min_dist - 1400) / 1800)
            else:
                print("pos  done")
                done = 1
                pos_dist_reward = -1
        if self.game_cnt > (
                self.max_step * (stage + 1 + (3 - self.state["alive enemy num"]))) or self.stay_cnt > 100 * (
                stage + 1 + (3 - self.state["alive enemy num"])):
            done = 1
            print("cnt done")
            # reward -= 5
        if self.state["HP"] < 1 or self.state["alive enemy num"] == 0:
            done = 1
            print("hp done")
        if done == 1:
            self.state["game over"] = np.array([1])
        # print(dist)

        reward += gun_reward + enemy_reward + helmet_reward + vest_reward + pos_dist_reward \
                  + HP_reward + damage_reward + ammo_reward + guide_reward - self.stay_cnt * 0.002 - self.game_cnt * 0.001

        # reward += gun_reward + enemy_reward + helmet_reward + vest_reward + pos_dist_reward_\
        #           + HP_reward + damage_reward + ammo_reward - self.stay_cnt * 0.002 - self.game_cnt * 0.001

        reward = reward.__float__()
        # print("reward: ", reward, "demage: ", damage_reward, "pos_dist_reward: ", pos_dist_reward,
        # " enemy_dist_reward: ", enemy_dist_reward)
        reward_delta = reward - self.reward + enemy_dist_reward
        # reward_delta = reward - self.reward
        # if abs(reward_delta) < 1:
        #     reward_delta *= 10
        self.reward = reward
        return reward_delta, done

    def close(self):
        self.reset()

    def step(self, action):
        action_done = self._do_action(action)
        if not action_done:
            return None
        self.game_cnt += 1
        # obs, detect_info = self._get_obs()
        obs = self._get_obs()
        if obs is None:
            return None
        reward, done = self._compute_reward()
        print("action: ", action, "  reward: ", reward)
        # return obs, self.detect_info, reward, done, self.state
        return (obs, reward, done, self.state)

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
            "guide": np.zeros(1),
        }
        msg = MSG()
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_move([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).to_bytes()
        reply = self.req.make_request(msg)

        time.sleep(10.0)
        action_done = self._do_action(np.zeros(3))
        if not action_done:
            return None
        obs = self._get_obs()
        if obs is None:
            return None
        return obs

    def _do_action(self, action) -> bool:
        action_cmd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # [0:move forward/backward, 1:move right/left, 2:yaw, 3:pitch, 4:crouch蹲,
        # 5:prone趴, 6:jump, 7:fire, 8:yaw_angle, 9:reload, 10:health, 11:restart]
        msg = MSG()
        msg.msg_type = MSGType.SEND_CTRL_CMD
        # if action[2] == 1:  # rotate right lightly
        #     # self.stay_cnt += 1
        #     action_cmd[8] = 1
        # elif action[2] == 2:  # rotate left lightly
        #     # self.stay_cnt += 1
        #     action_cmd[8] = -1
        # elif action[2] == 3:  # rotate right
        #     # self.stay_cnt += 1
        #     action_cmd[8] = 2
        # elif action[2] == 4:  # rotate left
        #     # self.stay_cnt += 1
        #     action_cmd[8] = -2
        # elif action[2] == 5:  # rotate right
        #     self.stay_cnt += 0.2
        #     action_cmd[8] = 4
        # elif action[2] == 6:  # rotate left
        #     self.stay_cnt += 0.2
        #     action_cmd[8] = -4
        # elif action[2] == 7:  # rotate right
        #     self.stay_cnt += 1
        #     action_cmd[8] = 10
        # elif action[2] == 8:  # rotate left
        #     self.stay_cnt += 1
        #     action_cmd[8] = -10
        # elif action[2] == 9:  # rotate right
        #     self.stay_cnt += 3
        #     action_cmd[8] = 30
        # elif action[2] == 10:  # rotate left
        #     self.stay_cnt += 3
        #     action_cmd[8] = -30
        # if action[3] == 1:
        #     if self.state["has gun"] and self.state["ammo"] > 0:
        #         action_cmd[7] = 1
        #     else:
        #         pass
        # if action[0] == 1:  # move forward
        #     action_cmd[0] = 1
        #
        # elif action[0] == 2:  # move backward
        #     action_cmd[0] = -1
        #
        # if action[1] == 1:  # move right
        #     action_cmd[1] = 1
        #
        # elif action[1] == 2:  # move left
        #     action_cmd[1] = -1
        #
        # if action[0] == 0 and action[1] == 0:
        #     self.stay_cnt += 2
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
            msg.msg_content = cmd_move(action_cmd).to_bytes()
            reply = self.req.make_request(msg)
            time.sleep(0.005)
        elif action[0] == 1:  # move forward
            action_cmd[0] = 1
            msg.msg_content = cmd_move(action_cmd).to_bytes()
            reply = self.req.make_request(msg)
            time.sleep(0.05)
        elif action[0] == 2:  # move backward
            action_cmd[0] = -1
            msg.msg_content = cmd_move(action_cmd).to_bytes()
            reply = self.req.make_request(msg)
            time.sleep(0.05)
        elif action[0] == 3:  # move right
            action_cmd[1] = 1
            msg.msg_content = cmd_move(action_cmd).to_bytes()
            reply = self.req.make_request(msg)
            time.sleep(0.05)
        elif action[0] == 4:  # move left
            action_cmd[1] = -1
            msg.msg_content = cmd_move(action_cmd).to_bytes()
            reply = self.req.make_request(msg)
            time.sleep(0.05)

        if reply is None:
            return False
        time.sleep(0.05)
        return True



if __name__ == "__main__":
    test = MaEnv(record=True, image_shape=(84, 168, 1), yolo_weight='../weights/best.pt')
    #req = RequestSender(addr="127.0.0.1", port=14514)
    while True:
        # msg = MSG()
        # msg.msg_type = MSGType.QUERY_POS
        # rep = req.make_request(msg)
        # print(rep)
        img1, b, c, d = test.step(np.zeros(3))
        if c:
            test.reset()

    # device = select_device("0")
    # yolo_weight = '../weights/best.pt'
    # yolo_model = attempt_load(yolo_weight, map_location=device)  # load FP32 yolo_model
    # pred = yolo_model(cv2.imread("D:\a.png"))[0]
    # print(pred)