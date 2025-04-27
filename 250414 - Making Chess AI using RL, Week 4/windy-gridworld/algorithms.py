from collections import defaultdict

import numpy as np


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, trials=10):
    """
    Q-learning algorithm with multiple trials for averaging.

    Args:
        env (WindyGridworld): The Windy Gridworld environment.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        episodes (int): Number of episodes.
        trials (int): Number of trials to average results.

    Returns:
        avg_rewards (list): Average cumulative rewards per episode over trials.
    """

    class Agent:
        def __init__(self, actions):
            self.actions = actions
            self.learning_rate = alpha
            self.discount_factor = gamma  # 감가율
            self.epsilon = epsilon  # 엡실론
            self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
        def learn(self, state, action, reward, next_state):
            q_1 = self.q_table[state][action]
            # 벨만 최적 방정식을 사용한 큐함수의 업데이트
            q_2 = reward + self.discount_factor * max(self.q_table[next_state])
            self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

        def get_action(self, state):
            action = 0
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                state_action = self.q_table[state]
                action = self.arg_max(state_action)
            return int(action)

        @staticmethod
        def arg_max(state_action):
            max_index_list = []
            max_value = state_action[0]
            for index, value in enumerate(state_action):
                if value > max_value:
                    max_index_list.clear()
                    max_value = value
                    max_index_list.append(index)
                elif value == max_value:
                    max_index_list.append(index)
            return np.random.choice(max_index_list)

    all_rewards = np.zeros((trials, episodes))

    env.reset()
    actions = ["up", "down", "left", "right"]
    agent = Agent(actions=list(range(4)))

    for trial in range(trials):
        for episode in range(episodes):
            state = env.reset()
            action = agent.get_action(state)
            total_reward = 0

            while True:
                # env.render()

                next_state, reward = env.step(state, actions[action])
                done = True if reward == 0 else False
                # next_state (tuple): Next state (x, y).
                # reward (int): Reward received.
                next_action = agent.get_action(next_state)

                agent.learn((state), action, reward, (next_state))
                total_reward += reward
                state = next_state
                action = next_action

                if done:
                    print("Episode : %d total reward = %f . " % (episode, total_reward))
                    all_rewards[trial, episode] = total_reward
                    break

    avg_rewards = np.mean(all_rewards, axis=0)

    return avg_rewards




def sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, trials=10):

    """
    SARSA algorithm with multiple trials for averaging.

    Args:
        env (WindyGridworld): The Windy Gridworld environment.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        episodes (int): Number of episodes.
        trials (int): Number of trials to average results.

    Returns:
        avg_rewards (list): Average cumulative rewards per episode over trials.
    """

    class Agent:
        def __init__(self, actions):
            self.actions = actions
            self.learning_rate = alpha
            self.discount_factor = gamma  # 감가율
            self.epsilon = epsilon  # 엡실론
            self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
        def learn(self, state, action, reward, next_state, next_action):
            q_1 = self.q_table[state][action]
            q_2 = reward + self.discount_factor * self.q_table[next_state][next_action]
            self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

        def get_action(self, state):
            action = 0
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                state_action = self.q_table[state]
                action = self.arg_max(state_action)
            return int(action)

        @staticmethod
        def arg_max(state_action):
            max_index_list = []
            max_value = state_action[0]
            for index, value in enumerate(state_action):
                if value > max_value:
                    max_index_list.clear()
                    max_value = value
                    max_index_list.append(index)
                elif value == max_value:
                    max_index_list.append(index)
            return np.random.choice(max_index_list)

    all_rewards = np.zeros((trials, episodes))

    env.reset()
    actions = ["up", "down", "left", "right"]
    agent = Agent(actions=list(range(4)))

    for trial in range(trials):
        for episode in range(episodes):
            state = env.reset()
            action = agent.get_action(state)
            total_reward = 0

            while True:
                # env.render()

                next_state, reward = env.step(state, actions[action])
                done = True if reward == 0 else False
                # next_state (tuple): Next state (x, y).
                # reward (int): Reward received.
                next_action = agent.get_action(next_state)

                agent.learn((state), action, reward, (next_state), next_action)
                total_reward += reward
                state = next_state
                action = next_action

                if done:
                    print("Episode : %d total reward = %f . " % (episode, total_reward))
                    all_rewards[trial, episode] = total_reward
                    break

    avg_rewards = np.mean(all_rewards, axis=0)

    return avg_rewards