import numpy as np
import torch
import torch.nn as nn

n_actions = 2

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def predict_probs(states, model):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :param model: torch model
    :returns: numpy array of shape [batch, n_actions]
    """
    with torch.no_grad():
        states = torch.tensor(states, dtype=torch.float32)
        logits = model(states)
        probs = torch.softmax(logits, dim=-1).numpy()
    return probs

def get_cumulative_rewards(rewards, gamma=0.99):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).
    :param rewards: list of immediate rewards
    :param gamma: discount factor
    :returns: list of cumulative rewards
    """
    cumulative_rewards = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        cumulative_rewards.append(G)
    cumulative_rewards.reverse()
    return cumulative_rewards

def get_loss(logits, actions, rewards, n_actions=n_actions, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    :param logits: model output logits [batch, n_actions]
    :param actions: taken actions [batch]
    :param rewards: rewards for the session [batch]
    :param n_actions: number of possible actions
    :param gamma: discount factor
    :param entropy_coef: coefficient for entropy regularization
    :returns: loss (torch tensor)
    """
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    actions_one_hot = to_one_hot(actions, n_actions)
    log_probs_for_actions = torch.sum(log_probs * actions_one_hot, dim=1)

    J_hat = torch.mean(log_probs_for_actions * cumulative_returns)

    entropy = -torch.sum(probs * log_probs, dim=-1).mean()

    loss = -J_hat - entropy_coef * entropy

    return loss