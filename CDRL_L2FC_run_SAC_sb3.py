from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, MultiBinary, Dict
import gym

from L2FC_env_gym import L2FC_env

environment=L2FC_env(algo_name, silent_mode=True)

from stable_baselines3 import PPO
from stable_baselines3 import SAC

def main():
    ob = environment.rotors_bc_reset('start')

    model = PPO("MlpPolicy", environment, verbose=1)
    model.learn(total_timesteps=50_000_000)

main()
