import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt


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
        )
        self.network2 = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),

        )
        self.b = nn.BatchNorm2d(32)

    def forward(self, x):
        x1 = self.network(x)
        x2 = self.network2(x)
        x3 = self.b(x1)
        x4 = self.b(x2)

        return x


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
d = a[2:4]
e = np.array([a[2],a[3]])
# the number of some of elements in a > 2
b = np.array(np.sum((a[2:4] > 2)))
print(b)

# a = Agent()
# x = torch.rand(1, 1, 84, 84)
# out = a(x)
# img_path = 'C:/Users/nly/Desktop/yolo_test_2/gamma/k (1).jpg'
# img = cv2.imread(img_path)
# cv2.imshow('img', img)
# cv2.waitKey(1000)
# img1 = (255 * (img / 255) ** 0.5).astype(np.uint8)
# cv2.imshow('img1', img1)
# cv2.waitKey(100000)

# a = torch.load('outputs/data_save/data.pt')

# print(a.shape)

# class agent(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(256, 128, batch_first=True)
#
#     def lstmm(self, x, hx, cx):
#         x, (hx, cx) = self.lstm(x, (hx, cx))
#         return x, (hx, cx)
#
# a = agent()
# x = torch.rand(12, 256)
# hx = torch.zeros(1, 128)
# cx = torch.zeros(1, 128)
# out, (hx, cx) = a.lstmm(x, hx, cx)
#
# xs = torch.rand(12, 1, 256)
# hxs = torch.zeros(1, 12, 128)
# cxs = torch.zeros(1, 12, 128)
# xs, (hxs, cxs) = a.lstm(xs, (hxs, cxs))
# print(xs.shape)


# hxs = torch.zeros((22, 1) + (512,))
#
# a = torch.tensor([[0.00000, 0.00000, 128.00000, 83.00000, 0.93799, 0.00000],
#                   [0.00000, 0.00000, 76.00000, 45.00000, 0.92969, 1.00000]])
# detect_matrix = torch.zeros(3, 84, 128)
# for *xyxy, conf, cls in reversed(a):
#     detect_matrix[int(cls), int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = 1
# a
#
# #
# device = torch.device("cuda:0")
# position = torch.zeros(1, 3)
# prev_position = np.random.rand(3)
# detect_info = np.random.rand(6)
# prev_detect_info = np.random.rand(6)
#
# a = torch.rand(1, 3)
# b = a[0][0]
#
# dict1 = {'position': position, 'prev_position': prev_position, 'detect_info': detect_info,
#          'prev_detect_info': prev_detect_info}
#
# # save dict1's value to torch tensor
# i = 0
# dict2 = []
# s = np.zeros((1, 18))
# for key, value in dict1.items():  # store dict1's value to s
#     if key == 'position':
#         s[0, i:i + 3] = value
#         i += 3
#     else:
#         s[0, i:i + len(value)] = value
#         i += len(value)
#
# np.concatenate((s, value), axis=0)
# s = torch.from_numpy(s).float().to(device)
# cc = [[]] * 5
# for i in range(5):
#     cc[i] = dict2
# print(cc)
# dd = torch.zeros(len(cc), len(cc[0]), len(cc[0][0]))
# for i in range(len(cc)):
#     dd[i] = cc[i][0]
# print(dd)
