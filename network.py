import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh,
                 output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_squeeze = output_squeeze
        
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x.squeeze() if self.output_squeeze else x

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, action_dim):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None):
        policy = Normal(self.mu(x), self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi

class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, action_dim):
        super(CategoricalPolicy, self).__init__()

        self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi

class BLSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, con_dim):
        super(BLSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims//2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dims, con_dim)
        nn.init.zeros_(self.linear.bias)

    def forward(self, seq, con=None):
        inter_states, _ = self.lstm(seq)
        logit_seq = self.linear(inter_states)
        self.logits = torch.mean(logit_seq, dim=1)
        policy = Categorical(logits=self.logits)
        label = policy.sample()
        logp = policy.log_prob(label).squeeze()
        if con is not None:
            logq = policy.log_prob(con).squeeze()
        else:
            logq = None

        return label, logq, logp

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space, hidden_dims=(64, 64), activation=torch.tanh):
        super(ActorCritic, self).__init__()

        if isinstance(action_space, Box):
            self.policy = GaussianPolicy(input_dim, hidden_dims, activation,  action_space.shape[0])
        elif isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(input_dim, hidden_dims, activation, action_space.n)

        self.value_f = MLP(layers=[input_dim] + list(hidden_dims) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value_f(x)

        return pi, logp, logp_pi, v

class Decoder(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dims=64):
        super(Decoder, self).__init__()
        self.policy = BLSTMPolicy(input_dim, hidden_dims, context_dim)

    def forward(self, seq, con=None):
        pred, logq, logp = self.policy(seq, con)
        return pred, logq, logp

