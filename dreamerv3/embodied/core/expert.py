import numpy as np


class ExpertAgent:
    def __init__(self, p_range, dx_range, dy_range, dz_range, dtheta_range):
        self.p_range = p_range
        self.dx_range = dx_range
        self.dy_range = dy_range
        self.dz_range = dz_range
        self.dtheta_range = dtheta_range

    def get_action_from_plan(self, plan):
        def get_unscaled_action(action, action_range):
            unscaled_action = (
                2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
            )
            return unscaled_action

        p = np.clip(plan[:, 0], self.p_range[0], self.p_range[1])
        dx = np.clip(plan[:, 1], self.dx_range[0], self.dx_range[1])
        dy = np.clip(plan[:, 2], self.dy_range[0], self.dy_range[1])
        dz = np.clip(plan[:, 3], self.dz_range[0], self.dz_range[1])
        dtheta = np.clip(plan[:, 4], self.dtheta_range[0], self.dtheta_range[1])

        unscaled_p = get_unscaled_action(p, self.p_range)
        unscaled_dx = get_unscaled_action(dx, self.dx_range)
        unscaled_dy = get_unscaled_action(dy, self.dy_range)
        unscaled_dz = get_unscaled_action(dz, self.dz_range)
        unscaled_dtheta = get_unscaled_action(dtheta, self.dtheta_range)

        unscaled_action = np.array(
            [unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta]
        )
        return unscaled_action.T

    def policy(self, plan_actions):
        output = {"action": np.array(plan_actions)[np.newaxis], "reset": False}
        return output, None
