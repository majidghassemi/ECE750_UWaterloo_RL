import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C
import highway_env  # noqa: F401

TRAIN = True

if __name__ == "__main__":
    env = gym.make("intersection-v0", render_mode="rgb_array")
    obs, info = env.reset()

    model = A2C(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        n_steps=10,
        gamma=0.99,
        gae_lambda=0.98,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
        use_rms_prop=True,
        normalize_advantage=True,
        verbose=1,
        tensorboard_log="intersection_A2C/",
        device="cuda"  # Ensure GPU usage
    )

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(0.6e5))
        model.save("intersection_A2C/model")
        del model

    model = A2C.load("intersection_A2C/model", env=env, device="cuda")
    env = RecordVideo(
        env, video_folder="intersection_A2C/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
