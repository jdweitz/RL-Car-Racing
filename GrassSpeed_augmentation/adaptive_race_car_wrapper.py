from gymnasium.envs.box2d.car_racing import CarRacing
from typing import Optional, Union
from gymnasium.error import DependencyNotInstalled, InvalidAction

import numpy as np


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

def is_in_grass(state_image):
    return (state_image[-20, 45, 0] != state_image[-20, 45, 1] or
            state_image[-20, 50, 0] != state_image[-20, 50, 1] or
            state_image[-29, 45, 0] != state_image[-29, 45, 1] or
            state_image[-29, 50, 0] != state_image[-29, 50, 1])

def corner_status_calculation(state_image):
    return [bool(state_image[-20, 45, 0] != state_image[-20, 45, 1]), 
            bool(state_image[-20, 50, 0] != state_image[-20, 50, 1]), 
            bool(state_image[-29, 45, 0] != state_image[-29, 45, 1]), 
            bool(state_image[-29, 50, 0] != state_image[-29, 50, 1])]



class CustomCarRacing(CarRacing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corner_status = [True, True, True, True]

    def corner_status_calculation(self, state_image):
        return [bool(state_image[-20, 45, 0] != state_image[-20, 45, 1]), 
                bool(state_image[-20, 50, 0] != state_image[-20, 50, 1]), 
                bool(state_image[-29, 45, 0] != state_image[-29, 45, 1]), 
                bool(state_image[-29, 50, 0] != state_image[-29, 50, 1])]
    
    def step(self, action: Union[ np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        self.corner_status = self.corner_status_calculation(self.state)
        corner_sum = sum(self.corner_status)

        step_reward = 0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1 + 0.08 * corner_sum
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}


class Late_game_CustomCarRacing(CarRacing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corner_status = [True, True, True, True]

    def corner_status_calculation(self, state_image):
        return [bool(state_image[-20, 45, 0] != state_image[-20, 45, 1]), 
                bool(state_image[-20, 50, 0] != state_image[-20, 50, 1]), 
                bool(state_image[-29, 45, 0] != state_image[-29, 45, 1]), 
                bool(state_image[-29, 50, 0] != state_image[-29, 50, 1])]
    
    def step(self, action: Union[ np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        self.corner_status = self.corner_status_calculation(self.state)
        corner_sum = sum(self.corner_status)

        step_reward = 0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1 + 0.025 * corner_sum
            if action[1] > 0.3:
                self.reward += action[1]*0.2
            # self.reward -= 0.1 + 0.08 * corner_sum
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}


class Late_game_CustomCarRacing2(CarRacing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corner_status = [True, True, True, True]

    def corner_status_calculation(self, state_image):
        return [bool(state_image[-20, 45, 0] != state_image[-20, 45, 1]), 
                bool(state_image[-20, 50, 0] != state_image[-20, 50, 1]), 
                bool(state_image[-29, 45, 0] != state_image[-29, 45, 1]), 
                bool(state_image[-29, 50, 0] != state_image[-29, 50, 1])]
    
    def step(self, action: Union[ np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        self.corner_status = self.corner_status_calculation(self.state)
        corner_sum = sum(self.corner_status)

        step_reward = 0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1 + 0.025 * corner_sum
            true_speed = np.sqrt(
                np.square(self.car.hull.linearVelocity[0])
                + np.square(self.car.hull.linearVelocity[1])
            )
            # print("true_speed: " , true_speed)
            if true_speed > 50:
                self.reward += 0.005 * (true_speed - 50)
            # self.reward -= 0.1 + 0.08 * corner_sum
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    

    
