import numpy as np

class rlalgorithm:
    def __init__(self, actions, learning_rate=0.1, gamma=0.9, epsilon=0.2, min_learning_rate=0.001, min_epsilon=0.01, decay_rate=0.99, *args, **kwargs):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_lr = min_learning_rate
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_table = {}
        self.display_name = "SARSA"

    def choose_action(self, observation, **kwargs):
        if observation not in self.q_table:
            self.q_table[observation] = np.zeros(len(self.actions))

        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            action = np.argmax(self.q_table[observation])
            return action

    def learn(self, s, a, r, s_, **kwargs):
        if s not in self.q_table:
            self.q_table[s] = np.zeros(len(self.actions))
        if s_ not in self.q_table:
            self.q_table[s_] = np.zeros(len(self.actions))
        
        a_ = self.choose_action(s_)
        
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * self.q_table[s_][a_]
        self.q_table[s][a] += self.lr * (q_target - q_predict)
        
        self.epsilon_decay()
        self.learning_rate_scheduler()
        
        return s_, a_
    
    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
    
    def learning_rate_scheduler(self):
        self.lr = max(self.min_lr, self.lr * self.decay_rate)
