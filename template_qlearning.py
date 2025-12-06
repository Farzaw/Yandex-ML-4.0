import numpy as np
import random
from collections import defaultdict


def my_softmax(values: np.ndarray, T=1.):
    max_val = np.max(values)
    exp_values = np.exp((values - max_val) / T)
    probas = exp_values / np.sum(exp_values)
    assert probas is not None
    return probas


class QLearningAgent:
    def __init__(self, alpha, discount, get_legal_actions, temp=1.):
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.discount = discount
        self.temp = temp

    def get_qvalue(self, state, action):
        """Returns Q(state,action)"""
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """Sets the Qvalue for [state,action] to the given value"""
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute V(s) = max_over_action Q(state,action) over possible actions.
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0
        value = max(self.get_qvalue(state, action) for action in possible_actions)
        assert value is not None
        return value

    def update(self, state, action, reward, next_state):
        """
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        gamma = self.discount
        learning_rate = self.alpha
        current_q = self.get_qvalue(state, action)
        next_value = self.get_value(next_state)
        qvalue = (1 - learning_rate) * current_q + learning_rate * (reward + gamma * next_value)
        assert qvalue is not None
        self.set_qvalue(state, action, qvalue)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        q_values = [self.get_qvalue(state, action) for action in possible_actions]
        best_action = possible_actions[np.argmax(q_values)]
        assert best_action is not None
        return best_action

    def get_softmax_policy(self, state):
        """
        Compute all actions probabilities using softmax policy.
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        q_values = np.array([self.get_qvalue(state, action) for action in possible_actions])
        assert q_values is not None
        probabilities = my_softmax(q_values, T=self.temp)
        assert probabilities is not None
        return probabilities

    def get_action(self, state):
        """
        Compute the action to take in the current state using softmax policy.
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        action_probs = self.get_softmax_policy(state)
        chosen_action = np.random.choice(possible_actions, p=action_probs)
        assert chosen_action is not None
        return chosen_action


class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of q-learning functions to implement Expected Value SARSA.
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    """

    def get_value(self, state):
        """
        Returns Vpi for current state under the softmax policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0

        action_probs = self.get_softmax_policy(state)
        
        value = sum(prob * self.get_qvalue(state, action) for prob, action in zip(action_probs, possible_actions))
        assert value is not None

        return value
