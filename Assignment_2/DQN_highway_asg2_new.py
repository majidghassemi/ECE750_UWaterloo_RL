import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import highway_env  # noqa: F401


def train_env(env_name):
    env = gym.make(
        env_name,
        config={
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
        },
    )
    env.reset()
    return env


def test_env(env_name):
    env = train_env(env_name)
    env.unwrapped.config.update({"policy_frequency": 15, "duration": 20})
    env.reset()
    return env


if __name__ == "__main__":
    env_name = "highway-fast-v0" 
    # intersection-v0, racetrack-v0, highway-fast-v0

    # Train
    model = DQN(
        "CnnPolicy",
        DummyVecEnv([lambda: train_env(env_name)]),
        learning_rate=5e-4,
        buffer_size=25000,
        learning_starts=200,
        batch_size=64,
        gamma=0.85,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.7,
        verbose=1,
        tensorboard_log=f"{env_name}_cnn/",
    )
    model.learn(total_timesteps=int(2e5))
    model.save(f"{env_name}_cnn/model")

    # Record video
    model = DQN.load(f"{env_name}_cnn/model")

    env = DummyVecEnv([lambda: test_env(env_name)])
    video_length = 2 * env.envs[0].config["duration"]
    env = VecVideoRecorder(
        env,
        f"{env_name}_cnn/videos/",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="dqn-agent",
    )
    obs, info = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    env.close()
