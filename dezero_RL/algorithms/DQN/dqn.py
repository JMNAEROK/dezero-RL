import numpy as np
from dezero import Variable
from dezero import functions as F
from dezero import layers as L
from dezero.models import MLP, Model
from dezero.optimizers import SGD
from dezero.core import no_grad
from collections import deque
import random

'''states = Variable(np.random.randn(32, 4)) #batch size = 32
next_states = Variable(np.random.randn(32, 4))
rewards = Variable(np.random.randn(32,))

hidden_size = 100
action_size = 8
iters = 50_000
gamma = 0.9

model = MLP((hidden_size, action_size), activation=F.sigmoid)
optimizer = SGD(lr=0.01).setup(model)

actions = np.random.randint(0, 8, size=32)
print(actions, len(actions))

#특정 action Q값 선택 방식
batch_index = np.arange(32)
qs = model(states)
q = qs[batch_index, actions]

#target 계산 + gradient 차단
with no_grad():
    next_qs = model(next_states) # (32, 8)
    next_q = F.max(next_qs, axis=1) # (32,)
    target = rewards + gamma * next_q

print(target.shape)'''

class QNet(Model):
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

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class DQNAgent:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.qnet = QNet()

    def update(self, state, action, reward, next_state, done):
        with no_grad():
            next_qs = self.qnet(next_state)
            next_q = F.max(next_qs, axis=1)
            
            target = reward + self.gamma * (1 - done) * next_q


if __name__ == '__main__':
    reward = np.array([1.0, 5.0])
    done = np.array([0, 1])
    gamma = 0.9

    next_q = np.array([
        [1.0, 3.0, 2.0],
        [4.0, 2.0, 1.0]
    ])

    next_q_max = next_q.max(axis=1)

    target = reward + gamma * (1 - done) * next_q_max

    print(next_q_max)
    print(target)