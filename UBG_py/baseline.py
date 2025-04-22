from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from TPSgym.test_env import TestEnv
from TPSgym.easy_env import EasyEnv
from TPSgym.shoot_env import ShootEnv
import gym
from gym import spaces
import numpy as np
import time

env = DummyVecEnv(
        [
            lambda: Monitor(
                TestEnv(image_shape=(128, 256, 1))
            )
        ]
    )
env = VecTransposeImage(env)
# 导入模型
# model = PPO.load("best_model.zip", env=env)
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.0001,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=1000,
    learning_starts=1000,
    buffer_size=100000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=1000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=100000, tb_log_name="dqn_run_" + str(time.time()), **kwargs
)

# Save policy weights
model.save("dqn_policy")
