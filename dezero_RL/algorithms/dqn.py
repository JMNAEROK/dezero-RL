import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))

import gymnasium as gym
from dezero import Variable
from dezero import functions as F
from dezero import layers as L
from dezero.models import MLP, Model
from dezero.optimizers import SGD
from dezero.core import no_grad
from collections import deque
import random, numpy as np, cv2


class OnlineQNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(action_size)

    def forward(self, state):
        state = F.relu(self.l1(state))
        return self.l2(state)

    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data]) # (batch_size, C, H, W)
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data]) # (batch_size, C, H, W)
        done = np.array([x[4] for x in data]).astype(np.int32)
        
        return state, action, reward, next_state, done


def preprocess(frame):
    #frame shape: (400, 600, 3), dtype: uint8

    #RGB -> grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #84 x 84 resize
    resized = cv2.resize(
        gray,
        (84, 84),
        interpolation=cv2.INTER_AREA
    )

    #uint8 [0, 255] -> float32 [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    obs, info = env.reset()

    frame = env.render()
    frame = preprocess(frame)

    # obs, rewawrd, terminated, truncated, info = env.step(action)

    # frame = env.render()
    # next_state = preprocess(frame)


    print(frame.shape)
    print(frame.dtype)