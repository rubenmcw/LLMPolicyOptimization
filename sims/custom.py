import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import numpy as np
import imageio.v2 as imageio
from scipy.linalg import solve_continuous_are
from dm_control import mujoco


# ============================================================
# Parameters
# ============================================================

params = {
    "l1": 0.3,
    "l2": 0.2,
    "lc1": 0.22,
    "lc2": 0.15,
    "m1": 1.0,
    "m2": 0.5,
    "I1": 0.08,
    "I2": 0.02,
    "g": 9.81,
    "Q": np.diag([300, 3000, 20, 10]),
    "R": np.diag([1, 1]),
    "fc1": 0.38,
    "fc2": 0.18,
    "fv1": 0.03,
    "fv2": 0.01,
}


TORQUE_LIMIT = 2.0

# action = 1.0 means elbow_torque = FUNSEARCH_TORQUE_SCALE
FUNSEARCH_TORQUE_SCALE = 1.2

START_THETA1 = 0.0
START_THETA2 = 0.0
START_DTHETA1 = 0.0
START_DTHETA2 = 0.0

STEPS = 1800
DT = 0.01

DEBUG_PRINT_EVERY = 10



# ============================================================
# Basic helpers
# ============================================================

def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(a, b):
    """Smallest signed angle difference a - b, wrapped to [-pi, pi]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def clip_unit(u):
    if not np.isfinite(u):
        return 0.0
    return float(np.clip(u, -1.0, 1.0))


# ============================================================
# Swing-up heuristic
# ============================================================

def heuristic(obs):
    """Returns one scalar elbow action between -1 and 1.

    obs size is 4:
      obs[0] = theta1
      obs[1] = theta2
      obs[2] = dtheta1
      obs[3] = dtheta2
    """

    # Initialize persistent latch once
    if not hasattr(heuristic, "pumper"):
        heuristic.pumper = False

    x1 = obs[0]
    x2 = obs[1]
    x3 = obs[2]
    x4 = obs[3]

    x1w = angle_diff(x1, 0.0)

    abs_x1 = abs(x1w)
    abs_x3 = abs(x3)

    switch_theta = 0.90
    v_scale = 1.0

    # Mode 1: original rock-up

    if abs_x1 < switch_theta and not heuristic.pumper:
        phase = x3 + 1.2 * np.sin(x1w)

        action = -1.0 if phase >= 0.0 else 1.0

        action += -0.03 * x4 - 0.02 * x2
        action += -0.15 * np.sin(x1w) * x3

        heuristic.last_mode = "rock"
        return clip_unit(action)

    # Mode 2: shoulder pump

    heuristic.pumper = True

    if abs_x3 > 0.10:
        action = -np.tanh(x3 / v_scale)
    else:
        action = 0.0

    heuristic.last_mode = "pump"
    return clip_unit(action)

def funsearched_swingup_controller(x):
    """Elbow-only swing-up controller."""
    obs = np.asarray(x, dtype=float)

    action_raw = heuristic(obs)
    action = np.clip(action_raw, -1.0, 1.0)

    elbow_torque = FUNSEARCH_TORQUE_SCALE * action
    u = np.array([0.0, elbow_torque], dtype=float)
    u = np.clip(u, -TORQUE_LIMIT, TORQUE_LIMIT)

    mode = getattr(heuristic, "last_mode", "swing")

    return u, float(action_raw), float(action), float(elbow_torque), mode


# ============================================================
# MuJoCo model
# ============================================================

def make_xml(p):
    return f"""
<mujoco model="custom_double_pendulum">
  <option timestep="{DT}" gravity="0 0 -{p["g"]}" integrator="RK4"/>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.02 0.02 0.06" rgb2="0.00 0.00 0.00"
             width="512" height="512"/>
    <texture name="checker" type="2d" builtin="checker"
             rgb1="0.25 0.25 0.25" rgb2="0.10 0.10 0.10"
             width="512" height="512"/>
    <material name="checker_mat" texture="checker"
              texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>

  <default>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0 -2 3" dir="0 1 -1"/>
    <camera name="fixed" mode="fixed"
            pos="0 -2.6 0.45"
            xyaxes="1 0 0 0 0 1"
            fovy="32"/>

    <geom name="floor" type="plane" pos="0 0 -0.15"
          size="2 2 0.01" material="checker_mat"/>

    <body name="link1" pos="0 0 0.45">
      <joint name="shoulder" type="hinge" axis="0 1 0"
             damping="{p["fv1"]}" frictionloss="{p["fc1"]}"/>
      <inertial pos="0 0 -{p["lc1"]}" mass="{p["m1"]}"
                diaginertia="{p["I1"]} {p["I1"]} {max(1e-4, p["I1"] * 0.02)}"/>
      <geom name="rod1" type="capsule" fromto="0 0 0 0 0 -{p["l1"]}"
            size="0.018" rgba="0.2 0.4 1 1"/>

      <body name="link2" pos="0 0 -{p["l1"]}">
        <joint name="elbow" type="hinge" axis="0 1 0"
               damping="{p["fv2"]}" frictionloss="{p["fc2"]}"/>
        <inertial pos="0 0 -{p["lc2"]}" mass="{p["m2"]}"
                  diaginertia="{p["I2"]} {p["I2"]} {max(1e-4, p["I2"] * 0.02)}"/>
        <geom name="rod2" type="capsule" fromto="0 0 0 0 0 -{p["l2"]}"
              size="0.015" rgba="1 0.4 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1"
           ctrllimited="true" ctrlrange="-{TORQUE_LIMIT} {TORQUE_LIMIT}"/>
    <motor name="elbow_motor" joint="elbow" gear="1"
           ctrllimited="true" ctrlrange="-{TORQUE_LIMIT} {TORQUE_LIMIT}"/>
  </actuator>
</mujoco>
"""


def get_raw_state(physics):
    return np.array([
        physics.data.qpos[0],
        physics.data.qpos[1],
        physics.data.qvel[0],
        physics.data.qvel[1],
    ], dtype=float)


def get_state(physics):
    """Wrapped state for controller/reward."""
    return np.array([
        wrap_pi(physics.data.qpos[0]),
        wrap_pi(physics.data.qpos[1]),
        physics.data.qvel[0],
        physics.data.qvel[1],
    ], dtype=float)


# ============================================================
# LQR around upright
# ============================================================

def make_lqr_gain(p):
    l1, lc1, lc2 = p["l1"], p["lc1"], p["lc2"]
    m1, m2 = p["m1"], p["m2"]
    I1, I2 = p["I1"], p["I2"]
    g = p["g"]
    Q = p["Q"]
    R = p["R"]

    M = np.array([
        [I1 + m1 * lc1**2 + m2 * l1**2, m2 * l1 * lc2],
        [m2 * l1 * lc2, I2 + m2 * lc2**2],
    ])

    G = np.array([
        (m1 * lc1 + m2 * l1) * g,
        m2 * lc2 * g,
    ])

    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0

    Ag = -np.linalg.inv(M) @ G.reshape(2, 1)
    A[2, 0] = Ag[0, 0]
    A[3, 0] = Ag[1, 0]

    B = np.zeros((4, 2))
    B[2:, :] = np.linalg.inv(M)

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    return K


K_LQR = make_lqr_gain(params)


def lqr_error_state(x):
    """Error around upright [pi, 0, 0, 0]."""
    theta1, theta2, dtheta1, dtheta2 = x

    return np.array([
        angle_diff(theta1, np.pi),
        angle_diff(theta2, 0.0),
        dtheta1,
        dtheta2,
    ], dtype=float)


def lqr_catch_controller(x):
    err = lqr_error_state(x)
    u = -K_LQR @ err
    return np.clip(u, -TORQUE_LIMIT, TORQUE_LIMIT)


def lqr_enter_condition(x):
    err = lqr_error_state(x)

    theta1_err = abs(err[0])
    theta2_err = abs(err[1])
    dtheta1 = abs(err[2])
    dtheta2 = abs(err[3])

    return (
        theta1_err < np.deg2rad(35.0)
        and theta2_err < 2.50
        and dtheta1 < 14.0
        and dtheta2 < 65.0
    )


def lqr_stay_condition(x):
    err = lqr_error_state(x)

    theta1_err = abs(err[0])
    theta2_err = abs(err[1])
    dtheta1 = abs(err[2])
    dtheta2 = abs(err[3])

    return (
        theta1_err < np.deg2rad(60.0)
        and theta2_err < 3.14
        and dtheta1 < 18.0
        and dtheta2 < 75.0
    )

# ============================================================
# Reward/debug helpers
# ============================================================

def reward_like_height(x):
    theta1, theta2, _, _ = x

    l1 = params["l1"]
    l2 = params["l2"]

    tip_z = -l1 * np.cos(theta1) - l2 * np.cos(theta1 + theta2)

    height_reward = (tip_z + l1 + l2) / (2.0 * (l1 + l2))
    height_reward = np.clip(height_reward, 0.0, 1.0)

    extension_reward = np.exp(-angle_diff(theta2, 0.0) ** 2)

    reward = 0.8 * height_reward + 0.2 * height_reward * extension_reward

    if lqr_enter_condition(x):
        reward += 5.0

    return float(reward)


# ============================================================
# Main sim
# ============================================================

def main():
    xml = make_xml(params)
    physics = mujoco.Physics.from_xml_string(xml)

    physics.data.qpos[:] = [START_THETA1, START_THETA2]
    physics.data.qvel[:] = [START_DTHETA1, START_DTHETA2]
    physics.forward()

    heuristic.last_mode = "rock"

    frames = []
    total_reward = 0.0

    swing_steps = 0
    lqr_steps = 0
    lqr_entries = 0

    lqr_active = False

    raw_actions = []
    clipped_actions = []
    elbow_torques = []
    shoulder_torques = []

    for i in range(STEPS):
        x = get_state(physics)
        raw_x = get_raw_state(physics)

        if lqr_active:
            lqr_active = lqr_stay_condition(x)
        else:
            if lqr_enter_condition(x):
                lqr_active = True
                lqr_entries += 1

        if lqr_active:
            u = lqr_catch_controller(x)
            raw_action = np.nan
            clipped_action = np.nan
            elbow_torque = u[1]
            mode = "lqr"
            lqr_steps += 1
        else:
            u, raw_action, clipped_action, elbow_torque, swing_mode = funsearched_swingup_controller(x)
            mode = swing_mode
            swing_steps += 1

        physics.data.ctrl[:] = u
        physics.step()

        total_reward += reward_like_height(get_state(physics))

        raw_actions.append(raw_action)
        clipped_actions.append(clipped_action)
        elbow_torques.append(u[1])
        shoulder_torques.append(u[0])

        if DEBUG_PRINT_EVERY and i % DEBUG_PRINT_EVERY == 0:
            err = lqr_error_state(x)

            print(
                f"i={i:04d} mode={mode:5s} "
                f"x=[{x[0]: .3f}, {x[1]: .3f}, {x[2]: .3f}, {x[3]: .3f}] "
                f"err=[{err[0]: .3f}, {err[1]: .3f}, {err[2]: .3f}, {err[3]: .3f}] "
                f"raw_q=[{raw_x[0]: .3f}, {raw_x[1]: .3f}] "
                f"u=[{u[0]: .3f}, {u[1]: .3f}] "
                f"raw={raw_action: .3f} clip={clipped_action: .3f} "
                f"shoulder={u[0]: .3f} elbow={u[1]: .3f}"
            )

        if i % 2 == 0:
            frames.append(physics.render(height=480, width=640, camera_id=0))

    out_name = "custom_double_pendulum_lqr_handoff.mp4"
    imageio.mimsave(out_name, frames, fps=30)

    raw_arr = np.asarray(raw_actions, dtype=float)
    clip_arr = np.asarray(clipped_actions, dtype=float)
    elbow_arr = np.asarray(elbow_torques, dtype=float)
    shoulder_arr = np.asarray(shoulder_torques, dtype=float)

    print(f"Saved video: {out_name}")
    print(f"Total custom reward: {total_reward:.3f}")
    print(f"Swing steps: {swing_steps}")
    print(f"LQR steps: {lqr_steps}")
    print(f"LQR entries: {lqr_entries}")
    print(f"Final wrapped state [theta1, theta2, dtheta1, dtheta2]: {get_state(physics)}")
    print(f"Final raw state     [theta1, theta2, dtheta1, dtheta2]: {get_raw_state(physics)}")

    if np.all(np.isnan(raw_arr)):
        print("Heuristic raw action min/max: nan nan")
        print("Heuristic clipped action min/max: nan nan")
    else:
        print("Heuristic raw action min/max:", np.nanmin(raw_arr), np.nanmax(raw_arr))
        print("Heuristic clipped action min/max:", np.nanmin(clip_arr), np.nanmax(clip_arr))

    print("Shoulder torque min/max:", np.nanmin(shoulder_arr), np.nanmax(shoulder_arr))
    print("Elbow torque min/max:", np.nanmin(elbow_arr), np.nanmax(elbow_arr))

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()