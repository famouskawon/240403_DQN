import numpy as np
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
state = env.reset()[0]
done = False

while not done:
    env.render() # 시각화
    action = np.random.choice([0,1]) # 행동 선택 무작위
    
    # 다음 상태 / 보상 / 목표상태 도달여부 / MDP 범위 밖의 종료 조건 충족 여부(시간 초과 등))
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"next_state, reward, terminated, truncated, info : {next_state}, {reward}, {terminated}, {truncated}, {info}")
    
    done = terminated | truncated # 둘 중 하나가 True이면 종료
env.close()

# <바닥부터 시작하는 딥러닝 4>