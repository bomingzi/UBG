import torch
from pathlib import Path
import os
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import cv2

load_path = 'outputs/pretrain_data.npz'
model_path = 'outputs/pretrain_model.pth'
data = np.load(load_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, bias_const)
        elif 'weight' in name:
            nn.init.orthogonal_(param, std)
    return layer


use_orthogonal_init = True


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


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
            layer_init(nn.Linear(512, 9), std=0.01),
        )
        # self.critic = nn.Sequential(
        #     # layer_init(nn.Linear(512, 64)),
        #     # nn.ReLU(),
        #     layer_init(nn.Linear(512, 1), std=1),
        # )

    def forward(self, s):
        s = s.permute(0, 3, 1, 2)
        s = torch.divide(s, torch.tensor(255.0))
        s = self.network(s)
        a = self.actor(s)
        a = torch.softmax(a, dim=1)
        # v = self.critic(s)
        return a


actions_data = data['a']
actions_data = one_hot(actions_data, 9)
obs_data = data['s']

# obs_data_ = np.zeros((obs_data.shape[0], 128, 256, 1))
# for i in range(obs_data.shape[0]):  # resize to 128*256
#     temp = obs_data[i, :, :, 0].astype(np.float64)
#     obs_data_[i, :, :, 0] = cv2.resize(temp, (256, 128))
# obs_data = obs_data_


# detect_data = data['detect_s']
actions_data_ = torch.tensor(actions_data)
obs_data_ = torch.tensor(obs_data)
# detect_data_ = torch.tensor(detect_data)
train_dataset = Data.TensorDataset(obs_data_.to(torch.float32),
                                   actions_data_.to(torch.float32),
                                   # detect_data_.to(torch.float32)
                                   )
loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

model = Agent().to(device)

model_load_path = 'outputs/ppo_model84x168.pt'
load_model = False
if load_model and os.path.exists(model_load_path):
    model_path = 'outputs/ppo_model_84x168.pt'
    checkpoint = torch.load(model_load_path)
    model.actor.load_state_dict(checkpoint['actor'])
    # model.critic.load_state_dict(checkpoint['critic'])
    model.conv_s.load_state_dict(checkpoint['conv_s'])
    # model.actor_detect.load_state_dict(checkpoint['actor_detect'])
    # model.critic_detect.load_state_dict(checkpoint['critic_detect'])
    # model.linear_cat.load_state_dict(checkpoint['linear_cat'])
    # model.lstm.load_state_dict(checkpoint['lstm'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print('load model from {}'.format(model_load_path))

criterion = nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)

for epoch in range(5000):
    train_right = []

    for batch_idx, (o, target) in enumerate(loader):
        o = o.to(device)
        target = target.to(device)
        # detect_o = detect_o.to(device)
        model.train()
        # output = model(o, detect_o)
        output = model(o)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss = {}'.format(loss.item()))
        if batch_idx % 100 == 0:
            model.eval()
            val_right = []
            for (o, target) in loader:
                o = o.to(device)
                target = target.to(device)
                # detect_o = detect_o.to(device)
                output = model(o)
                output = torch.argmax(output, dim=1)
                target = torch.argmax(target, dim=1)
                acc = torch.sum(output == target).item() / len(target)
                val_right.append(acc)
            print('epoch = {}, val acc = {}'.format(epoch, np.mean(val_right)))

save_state = {
    'actor': model.actor.state_dict(),
    # 'critic': checkpoint['critic'],
    'network': model.network.state_dict(),
    # 'actor_detect': model.actor_detect.state_dict(),
    # 'critic_detect': checkpoint['critic_detect'],
    # 'linear_cat': model.linear_cat.state_dict(),
    # "lstm": model.lstm.state_dict(),
    # 'optimizer': optimizer.state_dict(),
}
torch.save(save_state, model_path)
print('model saved')
