import os

# Must be set before importing dm_control.
# This lets MuJoCo render video in Docker/headless mode.
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import numpy as np
import imageio.v2 as imageio
from dm_control import suite


def concatenate_obs(time_step, obs_spec):
    return np.concatenate([
        time_step.observation[k].ravel()
        for k in obs_spec
    ])


def initialize_for_elbow_swingup(env):
    """
    Start near the hanging/downward configuration, but slightly perturbed.

    The small offset matters. If we initialize perfectly symmetrically with
    zero velocity, the system can look frozen, especially with zero action.
    """
    env.physics.named.data.qpos["shoulder"][0] = np.pi + 0.0
    env.physics.named.data.qpos["elbow"][0] = 0

    env.physics.named.data.qvel["shoulder"][0] = 0.0
    env.physics.named.data.qvel["elbow"][0] = 0.0

  
  
    env.physics.forward()


def heuristic(obs: np.ndarray) -> float:
    """Returns an action between -1 and 1.
    obs size is 6.
    """
    x1 = np.arctan2(-obs[1], obs[0])
    x2 = np.arctan2(-obs[2], obs[3])
    x3 = obs[4]
    x4 = obs[5]

    action = -0.168 * x2 + 0.096 * x4

    near_upright = (np.pi - 0.55 <= abs(x1) <= np.pi + 0.25) and (abs(x2) < 0.55)

    if not near_upright:
        pump_gain = 0.075
        if abs(x1) > np.pi - 0.75:
            pump_gain = 0.035

        action += -pump_gain * np.sin(x1) * x3
        action += 0.108 * abs(x1)
    else:
        action += 0.010 * x3 - 0.140 * x4 - 0.050 * x2

    return action



def main():
    env = suite.load(domain_name="acrobot", task_name="swingup")

    obs_spec = env.observation_spec()
    action_spec = env.action_spec()

    print("Observation spec:")
    print(obs_spec)
    print()
    print("Action spec:")
    print(action_spec)
    print()

    time_step = env.reset()
    initialize_for_elbow_swingup(env)

    # Refresh observation after manually changing qpos/qvel.
    zero_action = np.zeros(action_spec.shape, dtype=np.float32)
    time_step = env.step(zero_action)

    frames = []
    actions = []
    rewards = []

    total_reward = 0.0
    num_steps = 1000

    for i in range(num_steps):
        obs = concatenate_obs(time_step, obs_spec)

        u = heuristic(obs)
        u = np.clip(u, action_spec.minimum[0], action_spec.maximum[0])

        action = np.array([u], dtype=np.float32)

        time_step = env.step(action)

        reward = 0.0 if time_step.reward is None else float(time_step.reward)
        total_reward += reward

        actions.append(float(u))
        rewards.append(reward)

        # Render every other step so the video is smaller.
        if i % 2 == 0:
            frame = env.physics.render(height=480, width=640, camera_id=0)
            frames.append(frame)

    out_name = "acrobot_elbow_swingup_test1.mp4"
    imageio.mimsave(out_name, frames, fps=30)

    print(f"Saved video: {out_name}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Mean reward: {np.mean(rewards):.6f}")
    print(f"Action min/max: {np.min(actions):.3f}, {np.max(actions):.3f}")

    # Avoid noisy dm_control/OSMesa shutdown warnings after the video is saved.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()