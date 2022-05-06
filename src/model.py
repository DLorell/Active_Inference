import torch.nn as nn
import torch
import numpy as np

from src.constants import DEVICE

NUM_NEURONS = 20
EPSILON = 1e-8


class TransitionModel(nn.Module):
    """Representing P(s_t | s_(t-1) , a_(t-1)) state transition probability
    given the previous state and previous action."""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.linear_stack = nn.Sequential(
            nn.Linear(self.state_dim+self.action_dim, NUM_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(NUM_NEURONS, NUM_NEURONS),
            nn.LeakyReLU(),
            #nn.Linear(NUM_NEURONS, NUM_NEURONS),
            #nn.LeakyReLU(),
            nn.Linear(NUM_NEURONS, 2*self.state_dim)
        )
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_stack(x).reshape((-1, 2, self.state_dim))
        #mu = self.leaky_relu(x[:, 0, :])
        #var = self.relu(x[:, 1, :])
        #x = torch.stack((mu, var), dim=1)
        mu = self.sigmoid(x[:, 0, :])
        var = self.sigmoid(x[:, 1, :])
        x = torch.stack((mu, var), dim=1)
        return x

    def sample_state(self, sufficient_statistics: torch.tensor) -> torch.tensor:
        mu = sufficient_statistics[:, 0]
        var = sufficient_statistics[:, 1]
        std_normal_noise = np.random.standard_normal(self.state_dim)

        var[var==0] = EPSILON if self.training else 0

        sampled_state = mu + torch.tensor(std_normal_noise.astype(np.float32)).to(DEVICE) * torch.sqrt(var)
        return sampled_state

    def best_guess_state(self, sufficient_statistics: torch.tensor) -> torch.tensor:
        mu = sufficient_statistics[:, 0]
        return mu


class PosteriorModel(nn.Module):
    """Representing Q(s_t | s_(t-1) , a_(t-1), o_t)) posterior probability
    given the previous state, previous action, AND current observation."""
    def __init__(self, state_dim: int, action_dim: int, observation_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.linear_stack = nn.Sequential(
            nn.Linear(self.state_dim+self.action_dim+self.observation_dim, NUM_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(NUM_NEURONS, NUM_NEURONS),
            nn.LeakyReLU(),
            #nn.Linear(NUM_NEURONS, NUM_NEURONS),
            #nn.LeakyReLU(),
            nn.Linear(NUM_NEURONS, 2*self.state_dim)
        )
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_stack(x).reshape((-1, 2, self.state_dim))
        # mu = self.leaky_relu(x[:, 0, :])
        # var = self.relu(x[:, 1, :])
        # x = torch.stack((mu, var), dim=1)
        mu = self.sigmoid(x[:, 0, :])
        var = self.sigmoid(x[:, 1, :])
        x = torch.stack((mu, var), dim=1)
        return x

    def sample_state(self, sufficient_statistics: torch.tensor) -> torch.tensor:
        mu = sufficient_statistics[:, 0]
        var = sufficient_statistics[:, 1]
        std_normal_noise = np.random.standard_normal(self.state_dim)

        var[var==0] = EPSILON if self.training else 0

        sampled_state = mu + torch.tensor(std_normal_noise.astype(np.float32)).to(DEVICE) * torch.sqrt(var)
        return sampled_state

    def best_guess_state(self, sufficient_statistics: torch.tensor) -> torch.tensor:
        mu = sufficient_statistics[:, 0]
        return mu


class LikelihoodModel(nn.Module):
    """Representing P(o_t | s_t) likelihood probability given the current
    state."""
    def __init__(self, state_dim: int, observation_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.linear_stack = nn.Sequential(
            nn.Linear(self.state_dim, NUM_NEURONS),
            nn.LeakyReLU(),
            nn.Linear(NUM_NEURONS, NUM_NEURONS),
            nn.LeakyReLU(),
            #nn.Linear(NUM_NEURONS, NUM_NEURONS),
            #nn.LeakyReLU(),
            nn.Linear(NUM_NEURONS, 2*self.observation_dim),
        )
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear_stack(x).reshape((-1, 2, self.observation_dim))
        mu = self.leaky_relu(x[:, 0, :])
        var = self.relu(x[:, 1, :])
        x = torch.stack((mu, var), dim=1)
        return x

    def sample_observation(self, sufficient_statistics: torch.tensor) -> torch.tensor:
        mu = sufficient_statistics[:, 0]
        var = sufficient_statistics[:, 1]
        std_normal_noise = np.random.standard_normal(self.observation_dim)

        var[var==0] = EPSILON if self.training else 0

        sampled_observation = mu + torch.tensor(std_normal_noise.astype(np.float32)).to(DEVICE) * torch.sqrt(var)
        return sampled_observation

    def best_guess_observation(self, sufficient_statistics: torch.tensor) -> torch.tensor:
        mu = sufficient_statistics[:, 0]
        return mu

    def invert_prob(self, distribution: torch.tensor, observation: torch.tensor):
        pass


class GenerativeModel:
    def __init__(self, state_dim: int, action_dim: int, observation_dim: int):
        super().__init__()
        self.prior = TransitionModel(state_dim, action_dim)
        self.posterior = PosteriorModel(state_dim, action_dim, observation_dim)
        self.likelihood = LikelihoodModel(state_dim, observation_dim)
        self.models = [self.prior, self.posterior, self.likelihood]

    def parameters(self):
        return list(self.prior.parameters()) + list(self.posterior.parameters()) + list(self.likelihood.parameters())

    def train(self):
        for m in self.models:
            m.train()

    def eval(self):
        for m in self.models:
            m.eval()

    def to(self, device: str) -> 'GenerativeModel':
        for m in self.models:
            m.to(device)
        return self

    def save_state_dict(self, path: str):
        torch.save(self.prior.state_dict(), path + "/prior.pth")
        torch.save(self.posterior.state_dict(), path + "/posterior.pth")
        torch.save(self.likelihood.state_dict(), path + "/likelihood.pth")

    def load_state_dict(self, path: str):
        self.prior.load_state_dict(torch.load(path + "/prior.pth"))
        self.posterior.load_state_dict(torch.load(path + "/posterior.pth"))
        self.likelihood.load_state_dict(torch.load(path + "/likelihood.pth"))
