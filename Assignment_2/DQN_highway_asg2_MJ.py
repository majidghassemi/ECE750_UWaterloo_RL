import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from torch import nn
import highway_env  # noqa: F401


TRAIN = True

# Define environment and model configurations
env_configs = {
    "highway-fast": {
        "env_id": "highway-fast-v0",
        "model_name": "highway_dqn/model_highway",
        "log_dir": "highway_dqn_fast/",
    },
    "intersection": {
        "env_id": "intersection-v0",
        "model_name": "intersection_dqn/model_intersection",
        "log_dir": "intersection_dqn/",
    },
    "race": {
        "env_id": "racetrack-v0", 
        "model_name": "race_dqn/model_race",
        "log_dir": "race_dqn/",
    }
}

# Common policy configurations
policy_kwargs = dict(
    net_arch=[512, 256, 128],
    activation_fn=nn.Tanh,
    # layer_norm=True,
    # ortho_init=True
)

# Training configuration
model_config = {
    "policy_kwargs": policy_kwargs,
    "learning_rate": 3e-4,
    "buffer_size": 25000,
    "learning_starts": 200,
    "batch_size": 64,
    "gamma": 0.925,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 250,
    "verbose": 1
}


# Function to train and save a model
def train_and_save_model(env, model_path, log_dir, timesteps=int(5e4)):
    model = DQN("MlpPolicy", env, tensorboard_log=log_dir, **model_config)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    return model


# Function to evaluate and record a trained model
def evaluate_and_record_model(model_path, env, video_folder, episodes=10):
    model = DQN.load(model_path, env=env)
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.unwrapped.config["simulation_frequency"] = 15

    for _ in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()


if __name__ == "__main__":
    for env_name, config in env_configs.items():
        # Initialize environment
        env = gym.make(config["env_id"], render_mode="rgb_array")

        # Train and save model if TRAIN
        if TRAIN:
            print(f"Training model for {env_name} environment.")
            train_and_save_model(env, config["model_name"], config["log_dir"])

        # Evaluate and record video
        print(f"Evaluating and recording model for {env_name} environment.")
        evaluate_and_record_model(config["model_name"], env, f"{config['log_dir']}videos")
