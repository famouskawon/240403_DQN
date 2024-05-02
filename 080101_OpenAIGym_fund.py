import gymnasium as gym
# import gym

env = gym.make('CartPole-v1')
# env = gym.make('CartPole-v0')
# 
# env = gym.make('CartPole-v0', render_mode='human')

state = env.reset()[0]
print(f'state: {state}')

action_space = env.action_space
print(f'action_space: {action_space}')

action = 0
next_state, reward, terminated, truncated, info = env.step(action)
print(f"next_state, reward, terminated, truncated, info : {next_state}, {reward}, {terminated}, {truncated}, {info}")

# <바닥부터 시작하는 딥러닝 4>