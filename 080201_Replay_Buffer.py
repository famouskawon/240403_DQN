from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen = buffer_size)
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
        
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):

    state = env.reset()[0]
    done = False

    while not done:
        action = 0
        
        # 다음 상태 / 보상 / 목표상태 도달여부 / MDP 범위 밖의 종료 조건 충족 여부(시간 초과 등))
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"next_state, reward, terminated, truncated, info : {next_state}, {reward}, {terminated}, {truncated}, {info}")
        
        done = terminated | truncated # 둘 중 하나가 True이면 종료
        
        replay_buffer.add(state, action, reward, next_state, done)
        sate = next_state

batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.get_batch()
print(f"batch_state.shape, batch_action.shape, batch_reward.shape, batch_next_state.shape, batch_done.shape: {batch_state.shape}, {batch_action.shape}, {batch_reward.shape}, {batch_next_state.shape}, {batch_done.shape}")