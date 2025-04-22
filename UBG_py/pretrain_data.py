import numpy as np
import argparse

from TPSgym.test_env import TestEnv
import cv2
import gym
from gym import spaces
import os

import torch

key_map = {
    ord("a"): 4,  #
    ord("s"): 2,  # backward
    ord("d"): 3,  #
    ord("w"): 1,  # forward
    ord("q"): 6,  # rotate left
    ord("e"): 5,  # rotate right
    ord("j"): 8,  #
}

if __name__ == '__main__':
    # yolo_device = "0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = TestEnv(image_shape=(84, 168, 1))
    s = []
    # detect_s = []
    a = []
    load_npz = True
    load_path = 'outputs/pretrain_data.npz'
    save_path = 'outputs/pretrain_data.npz'
    if load_npz and os.path.exists(load_path):
        data = np.load(load_path)
        s = data['s'].tolist()
        # detect_s = data['detect_s'].tolist()
        a = data['a'].tolist()

    cv2.namedWindow("bh")
    cv2.resizeWindow('bh', 100, 100)
    while True:
        key = cv2.waitKey(0)
        action = key_map[key]
        print(action)
        obs, _, _, _ = env.step(0)
        # detect_obs = env.detect_info.numpy()
        _, reward, done, info = env.step(action)

        s.append(obs)
        # detect_s.append(detect_obs)
        a.append(action)
        if done:
            env.reset()
            # np.savez(save_path, s=s, detect_s=detect_s, a=a)
            np.savez(save_path, s=s, a=a)
        else:
            _, _, _, _ = env.step(0)
