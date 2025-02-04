import random
from Worlds import GridWorld, LineWorld
from llm_supervision import llm_supervision


def q_learning(env: GridWorld | LineWorld, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    :param env:
    :param episodes:
    :param alpha:
    :param gamma:
    :param epsilon: exploration-exploitation parameter. The higher its value, the more exploration.
    :return:
    """
    Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}

    for episode in range(episodes):
        state = env.reset()
        done = False
        state_traj = []
        action_traj = []

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_action_space())  # Exploration
            else:
                action = max(Q[state], key=Q[state].get)  # Exploitation

            next_state, reward, done = env.step(action)
            # print(next_state, reward, done)

            best_next_action = max(Q[next_state], key=Q[next_state].get)

            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state

            state_traj.append(state)
            action_traj.append(action)

        # print(f"States trajectory: {state_traj}\nActions trajectory: {action_traj}")


    return Q


def q_learning_with_llm(env, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}
    history = []

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_action_space())  # Exploration
            else:
                action = max(Q[state], key=Q[state].get)  # Exploitation
                print(action, Q[state], max(Q[state].values()))

            supervised_action = llm_supervision(position=state, suggested_direction_untranslated=action,
                                                direction_certitude=max(Q[state].values()), history=history)

            print(supervised_action)

            next_state, reward, done = env.step(supervised_action)

            print(next_state, reward, done)

            best_next_action = max(Q[next_state], key=Q[next_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state

    return Q


if "__main__" == __name__:
    # 1D test
    terminal_state = 29  # Extrémité droite de la ligne
    line = LineWorld(size=terminal_state+1, goal_position=terminal_state, reward=10)
    q_table_line = q_learning(line, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1)
    # Print the learned Q-values
    for state, actions in q_table_line.items():
        print(f"State {(state)}: {actions}")

    # 2D test
    # terminal_states = ((0,6),(0,7),(0,8),(1,6),(1,7),(1,8))
    # enviro = GridWorld(size=10, goal_position=terminal_states, reward=10)
    #
    # # Train the agent
    # Q_table = q_learning(enviro, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1)
    #
    # # Print the learned Q-values
    # for state, actions in Q_table.items():
    #     print(f"State {state}: {actions}")