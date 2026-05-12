def funsearched_swingup_controller(x):
    theta1, theta2, dtheta1, dtheta2 = x

    x1 = wrap_pi(theta1)
    x2 = wrap_pi(theta2)
    x3 = dtheta1
    x4 = dtheta2

    switch_theta = 0.90

    if abs(x1) < switch_theta:
        # Rocking.
        phase = x3 + 1.2 * np.sin(x1)

        if phase >= 0.0:
            action = -1.0
        else:
            action = 1.0

        # Light damping. Too much damping traps it in rocking mode.
        action += -0.03 * x4 - 0.02 * x2

        # Small shoulder-energy encouragement.
        action += -0.15 * np.sin(x1) * x3

    else:
        # Shoulder pump.
        v_scale = 1.0

        if abs(x3) > 0.10:
            # Brutal version was:
            # action = -np.sign(x3)
            action = -np.tanh(x3 / v_scale)
        else:
            action = 0.0

    ### sim worked around .75 but walk this in on hardware
    elbow_torque_scale = 0.86

    ### if it damps instead of swinging just throw a minus sign on the torque.
    elbow_torque = elbow_torque_scale * np.clip(action, -1.0, 1.0)

    return [0.0, elbow_torque]