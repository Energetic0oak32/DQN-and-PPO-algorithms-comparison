from actions import Action


def simple_reward(action, dangers, collision):
    action = Action(action)

    if collision:
        return -1.0

    reward = 0.1
    front_danger = dangers["front"]

    if action == Action.FORWARD:
        if front_danger < 0.5:
            reward += 0.2
        elif front_danger < 0.7:
            reward += 0.05
        else:
            reward -= 0.3
    elif action == Action.STOP:
        reward -= 0.05
    elif action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        if front_danger > 0.6:
            reward += 0.15
        else:
            reward -= 0.05

    return reward

def advanced_reward(action, dangers, collision):
    action = Action(action)

    if collision:
        return -10.0

    reward = 0.05
    max_danger = dangers["max"]
    front_danger = dangers["front"]
    left_danger = dangers["left"]
    right_danger = dangers["right"]

    reward -= max_danger * 0.5
    reward -= front_danger * 0.8

    if action == Action.FORWARD:
        if front_danger < 0.3:
            reward += 0.6
        elif front_danger < 0.5:
            reward += 0.3
        elif front_danger < 0.7:
            reward += 0.1
        else:
            reward -= 0.5
    elif action == Action.STOP:
        reward -= 0.1
    elif action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        if front_danger > 0.6:
            reward += 0.4
            if (
                (action == Action.TURN_LEFT and right_danger > left_danger)
                or (action == Action.TURN_RIGHT and left_danger > right_danger)
            ):
                reward += 0.3
        else:
            reward -= 0.1

    return reward
