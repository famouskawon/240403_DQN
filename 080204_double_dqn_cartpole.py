import copy
from collections import deque
import random
import numpy as np
import gymnasium as gym # not using gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        # 경험 재생 버퍼에 경험 데이터 추가
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            # 데이터가 쌓이지 않았다면 바로 종료
            return

        # 미니배치 크기 이상이 쌓이면 미니배치 생성
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        
        qs = self.qnet(state) # qs([32,2]) | state([32,4])
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet(next_state) # next_qs([32,2])
        argmax_actions = next_qs.argmax(1)[0] # DQN: $\max_a Q(s_t+1,a;\theta_t^-)$ next_q([32])
        
        # next_q = next_qs.max(1)[0] # DQN: $\max_a Q(s_t+1,a;\theta_t^-)$ next_q([32])
        
        next_qs = self.qnet_target(next_state)
        next_q = next_qs[np.arange(self.batch_size),argmax_actions]
        
        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


episodes = 2000
sync_interval = 20
env = gym.make('CartPole-v1')
agent = DQNAgent()
reward_history = []

for episode in range(episodes):
    print("")
    print(f"Episode {episode} start!")
    state = env.reset()[0]
    #state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        print(".",end="")
        # next_state, reward, done, info = env.step(action)
        next_state, reward, terminated, truncated, info  = env.step(action)

        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # # added because WARN: You are calling 'step()' even though this environment has already returned terminated = True.
        # # You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.
    
    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("")
        print("episode :{}, total reward : {}".format(episode, total_reward))



import pickle
import os

# 파일 이름 생성 및 파일 존재 여부 확인
index = 0
filename = f'080205_CartpoleResults/Double_DQN_reward_history_{index}.pkl'
while os.path.exists(filename):
    index += 1  # 파일이 이미 존재하면 인덱스 증가
    filename = f'080205_CartpoleResults/Double_DQN_reward_history_{index}.pkl'  # 새로운 파일 이름 업데이트

# 파일이 존재하지 않으면, 새로운 인덱스를 사용해 파일 저장
with open(filename, 'wb') as f:
    pickle.dump(reward_history, f)

print(f'File saved: {filename}')
