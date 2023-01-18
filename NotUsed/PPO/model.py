import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, MultivariateNormal

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        current_rate = self.start ** (current_step // self.decay)
        if current_rate < self.end:
            return self.end
        else:
            return current_rate

class ConvNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()
        # actor

        self.fc1 = nn.Linear(state_dim, n_latent_var)
        self.fc2 = nn.Linear(n_latent_var, n_latent_var)
        self.action_layer = nn.Linear(n_latent_var, action_dim)
        self.value_layer = nn.Linear(n_latent_var, 1)

        self.cov_var = torch.full(size=(action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        x = self.fullyLayer(torch.from_numpy(state))
        action_probs = self.action_layer(x)
        dist = MultivariateNormal(action_probs, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print(log_prob, action)
        return action, log_prob

    def evaluate(self, state, action):
        # print(state.shape, action.shape)
        x = self.fullyLayer(state)
        action_probs = self.action_layer(x)
        # print(action_probs.requires_grad)
        dist = MultivariateNormal(action_probs, self.cov_mat)

        # print(dist)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(x)

        # print(action_logprobs.shape, state_value.shape, dist_entropy.shape)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def tester(self, state, action):
        # print(state.shape, action.shape)
        x = self.fullyLayer(torch.from_numpy(state))
        action_probs = self.action_layer(x)
        dist = MultivariateNormal(action_probs, self.cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(x)
        return action_logprobs, torch.squeeze(state_value), dist_entropy, action_probs, action_logprobs

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

    def fullyLayer(self, x):
        x = F.tanh(self.fc1(x))
        return F.tanh(self.fc2(x))