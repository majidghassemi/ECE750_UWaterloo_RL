import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401

def train_and_run(env_name, total_timesteps=15e4, n_cpu=6, batch_size=128):
    env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    
    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log=f"{env_name}_ppo/",
    )
    
    # Train the agent
    model.learn(total_timesteps=int(total_timesteps))
    # Save the agent
    model.save(f"{env_name}_ppo/model")
    
    # Load and run the trained model
    model = PPO.load(f"{env_name}_ppo/model")
    env = gym.make(env_name, render_mode="rgb_array")
    
    for _ in range(50):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

if __name__ == "__main__":
    env_name = "racetrack-v0"
    # intersection-v0, racetrack-v0, highway-fast-v0
    train_and_run(env_name)