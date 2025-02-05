import random
import numpy as np
import matplotlib.pyplot as plt
from Worlds import GridWorld, LineWorld
import llm_supervision as llm


def q_learning(env: GridWorld|LineWorld, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}
    iterations_by_episode = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        iteration = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_action_space())  # Exploration
            else:
                action = max(Q[state], key=Q[state].get)  # Exploitation

            next_state, reward, done = env.step(action)

            best_next_action = max(Q[next_state], key=Q[next_state].get)

            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state

            iteration += 1

        iterations_by_episode.append(iteration)

        # print(f"States trajectory: {state_traj}\nActions trajectory: {action_traj}")


    return Q, iterations_by_episode


def q_learning_with_llm(env: GridWorld|LineWorld, episodes=100, episodes_with_llm=5, alpha=0.1, gamma=0.9, epsilon=0.1):
    iterations_by_episode = []

    # llm_agent = llm.get_openai_client()

    traj_state = []
    traj_action = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        iteration = 0

        traj_state_episode = []
        traj_action_episode = []

        while not done:
            if episode < episodes_with_llm:  # LLM part
                action = random.choice(env.get_action_space())

                is_getting_closer = env.is_action_bringing_closer_to_goal(action)

                # context = llm.context_prompt_1d()
                prompt = llm.context_prompt_1d()
                prompt += llm.prompt_with_action_1d_and_proximity_cue(action, is_getting_closer)

                # response = llm.get_response_from_openai_prompt(llm_agent, prompt, context)
                response = llm.get_response_from_ollama_prompt(prompt)
                print(f"Épisode : {episode+1}.\nAction RL : {action}\nRéponse : {response}")

                action = env.get_action_from_str(response)

                next_state, reward, done = env.step(action)

                traj_state_episode.append(state)
                traj_action_episode.append(action)

                state = next_state

                iteration += 1

            else:
                if episode == episodes_with_llm:  # when the LLM is released
                    Q = {state: {action: -10 for action in env.get_action_space()} for state in env.get_state_space()}
                    for states, actions in zip(traj_state, traj_action):
                        for s, a in zip(states, actions):
                            Q[s][a] += 1 / episodes_with_llm

                    print_q_table(Q)

                # RL part
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(env.get_action_space())  # Exploration
                else:
                    action = max(Q[state], key=Q[state].get)  # Exploitation

                next_state, reward, done = env.step(action)

                best_next_action = max(Q[next_state], key=Q[next_state].get)

                Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

                state = next_state

                iteration += 1

        traj_state.append(traj_state_episode)
        traj_action.append(traj_action_episode)

        iterations_by_episode.append(iteration)

    return Q, iterations_by_episode


def print_q_table(q):
    for state, actions in q.items():
        print(f"State {(state)}: {actions}")


def plot_time_by_episode(time_by_episode, label):
    for times, l in zip(time_by_episode, label):
        plt.plot(times, label=l)
    plt.xlabel('Number of episode')
    plt.ylabel('Time spent in each episode')
    plt.legend(loc='best')
    plt.show()


if "__main__" == __name__:
    # 1D test
    terminal_state = 19  # Extrémité droite de la ligne
    line = LineWorld(size=terminal_state+1, goal_position=terminal_state, penalty=-1, reward=10)
    replicates = 30
    replicates_time = []
    for i in range(replicates):
        q_table_line, times_in_episodes = q_learning(line, episodes=60, alpha=0.1, gamma=0.9, epsilon=0.1)
        replicates_time.append(times_in_episodes)
    mean_times_in_episodes = np.mean(replicates_time, axis=0)


    q_table_with_llm, time_in_episodes_llm = q_learning_with_llm(line, episodes=60, episodes_with_llm=5, alpha=0.1,
                                                                 gamma=0.9, epsilon=0.1)

    plot_time_by_episode([mean_times_in_episodes, time_in_episodes_llm], label=["RL", "RL+LLM"])

    print(f"Q table without LLM:")
    print_q_table(q_table_line)
    #
    # print(f"Q table with LLM:")
    # print_q_table(q_table_with_llm)


    # 2D test
    # terminal_states = ((0,6),(0,7),(0,8),(1,6),(1,7),(1,8))
    # enviro = GridWorld(size=10, goal_position=terminal_states, reward=10)
    #
    # # Train the agent
    # Q_table, _ = q_learning(enviro, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1)
    #
    # # Print the learned Q-values
    # for state, actions in Q_table.items():
    #     print(f"State {state}: {actions}")
