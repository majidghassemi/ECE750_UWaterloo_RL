import numpy as np

class rlalgorithm:
    def __init__(self, actions, learning_rate=0.1, gamma=0.9, epsilon=0.2,
                 min_learning_rate=0.001, min_epsilon=0.01, decay_rate=0.99, *args, **kwargs):
        self.actions = actions 
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_lr = min_learning_rate
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_one = {}
        self.q_two = {}
        self.display_name = "Double Q-Learning"

    def choose_action(self, observation, **kwargs):
        if observation not in self.q_one:
            self.q_one[observation] = np.zeros(len(self.actions))
        if observation not in self.q_two:
            self.q_two[observation] = np.zeros(len(self.actions))

        if np.random.uniform() < self.epsilon:
            action_index = np.random.choice(len(self.actions))
        else:
            q_values_combined = self.q_one[observation] + self.q_two[observation]
            action_index = np.argmax(q_values_combined)
        return action_index

    def learn(self, s, a, r, s_, **kwargs):
        a_index = a 

        if s_ not in self.q_one:
            self.q_one[s_] = np.zeros(len(self.actions))
        if s_ not in self.q_two:
            self.q_two[s_] = np.zeros(len(self.actions))
        if s not in self.q_one:
            self.q_one[s] = np.zeros(len(self.actions))
        if s not in self.q_two:
            self.q_two[s] = np.zeros(len(self.actions))

        if np.random.rand() < 0.5:
            a_max = np.argmax(self.q_one[s_])
            q_target = r + self.gamma * self.q_two[s_][a_max]
            q_predict = self.q_one[s][a_index]
            self.q_one[s][a_index] += self.lr * (q_target - q_predict)
        else:
            a_max = np.argmax(self.q_two[s_])
            q_target = r + self.gamma * self.q_one[s_][a_max]
            q_predict = self.q_two[s][a_index]
            self.q_two[s][a_index] += self.lr * (q_target - q_predict)

        next_action = self.choose_action(s_)
        return s_, next_action

    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def learning_rate_scheduler(self):
        self.lr = max(self.min_lr, self.lr * self.decay_rate)
