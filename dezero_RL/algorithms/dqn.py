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

    #uint8 [0, 255] -> float32 [0.0, 1.0], not in the paper
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def get_action(qnet, state, epsilon=0.1, action_size=2): #epsilon greedy
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)

    state = np.expand_dims(state, axis=0) # (4, 84, 84) -> (1, 4, 84, 84)

    with no_grad():
        q = qnet(state)

    return int(q.data.argmax(axis=1)[0])


class QNet(Model):
    def __init__(self, action_size=2):
        super().__init__()

        self.conv1 = L.Conv2d(32, kernel_size=8, stride=4)
        self.conv2 = L.Conv2d(64, kernel_size=4, stride=2)
        self.conv3 = L.Conv2d(64, kernel_size=3, stride=1)

        self.fc1 = L.Linear(512)
        self.fc2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.reshape(x, (x.shape[0], -1))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    
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


def update_target(online_qnet, target_qnet):
    for online_param, target_param in zip(
        online_qnet.params(),
        target_qnet.params()
    ):
        target_param.data = online_param.data.copy()


def train_step(online_qnet, target_qnet, replay_buffer, optimizer, gamma=0.9):
    state, action, reward, next_state, done = replay_buffer.get_batch()

    q_values = online_qnet(state)
    batch_size = state.shape[0]

    q = q_values[np.arange(batch_size), action]

    with no_grad():
        next_q_values = target_qnet(next_state)
        next_q = np.max(next_q_values.data, axis=1)

        target = reward + gamma * (1 - done) * next_q

    target = Variable(target.astype(np.float32))

    #loss
    loss = F.mean_squared_error(q, target)

    #update
    online_qnet.cleargrads()
    loss.backward()
    optimizer.update()

    return float(loss.data)



if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()

    frames = deque(maxlen=4) #the paper uses 4 frames as an input
    
    replay_buffer = ReplayBuffer(
            buffer_size=5000,
            batch_size=32
        )
    
    online_qnet = QNet(action_size=2)
    target_qnet = QNet(action_size=2)

    optimizer = SGD(0.0001).setup(online_qnet)

    episodes = 1000
    max_steps = 1000
    total_steps = 0

    learning_start = 1000

    for episode in range(episodes):
        env.reset()

        frames.clear()
        frame = preprocess(env.render())

        for _ in range(4):
            frames.append(frame)

        state = np.stack(frames, axis=0)

        for step in range(max_steps): # 500 steps
            action = get_action(online_qnet, state)
            total_steps += 1

            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_frame = preprocess(env.render())
            frames.append(next_frame)
            next_state = np.stack(frames, axis=0)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            #training
            if len(replay_buffer) >= learning_start:
                loss = train_step(
                    online_qnet, 
                    target_qnet, 
                    replay_buffer, 
                    optimizer
                )

            if total_steps % 1000 == 0:
                update_target(online_qnet, target_qnet)

            if done:
                break