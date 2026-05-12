def funsearched_swingup_controller(x):
    theta1, theta2, dtheta1, dtheta2 = x

    # Persistent latch.
    # Once this becomes True, controller stays in pump mode.
    if not hasattr(funsearched_swingup_controller, "pumper"):
        funsearched_swingup_controller.pumper = False

    x1 = wrap_pi(theta1)
    x2 = wrap_pi(theta2)
    x3 = dtheta1
    x4 = dtheta2

    switch_theta = 0.90

    # Separate torque scales
    rock_elbow_torque_scale = 0.86
    pump_elbow_torque_scale = 0.86

    # Mode 1: rock-up starter

    if abs(x1) < switch_theta and not funsearched_swingup_controller.pumper:
        phase = x3 + 1.2 * np.sin(x1)

        if phase >= 0.0:
            action = -1.0
        else:
            action = 1.0

        # Light damping. Too much damping traps it in rocking mode.
        action += -0.03 * x4 - 0.02 * x2

        # Small shoulder-energy encouragement.
        action += -0.15 * np.sin(x1) * x3

        torque_scale = rock_elbow_torque_scale
        funsearched_swingup_controller.last_mode = "rock"


    # Mode 2: latched shoulder pump

    else:
        funsearched_swingup_controller.pumper = True

        v_scale = 1.0

        if abs(x3) > 0.10:

            action = -np.tanh(x3 / v_scale)
        else:
            action = 0.0

        torque_scale = pump_elbow_torque_scale
        funsearched_swingup_controller.last_mode = "pump"

    ### if it damps instead of swinging just throw a minus sign on the torque.
    elbow_torque = torque_scale * np.clip(action, -1.0, 1.0)

    return [0.0, elbow_torque]