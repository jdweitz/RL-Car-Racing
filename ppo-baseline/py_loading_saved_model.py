import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# import numpy as np
import pygame

env = gym.make('CarRacing-v2', render_mode='human')
env = DummyVecEnv([lambda: env])  # Wrap in a dummy vectorized environment

model = PPO.load("car_racing_ppo2", env=env)

episodes = 1
for episode in range(episodes):
    obs = env.reset()
    done = False
    total_rewards = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()
        total_rewards += rewards

        # Check for events, if escape is pressed, exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True
                print("Escape key pressed. Exiting...")
                break

    print(f"Total rewards for episode {episode + 1}: {total_rewards}")
env.close()
