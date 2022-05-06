import gym
import pygame
import torch
from gym.utils import play
import numpy as np
import matplotlib.pyplot as plt

from src.data import MountainCarData, MountainCar
from src.model import GenerativeModel

from src.constants import DEVICE


STATE_DIMENSION = 4
ACTION_DIMENSION = 1
OBSERVATION_DIMENSION = 2

switch_amount = 30
next_switch = switch_amount
direction = 1


if __name__ == "__main__":
    action_seq = []
    for i in range(300):
        #if i >= next_switch:
        #    next_switch = next_switch + switch_amount
        #    direction = -1 * direction
        #action = direction * (i - next_switch)/switch_amount
        action = MountainCar(step_lim=np.inf).sample_action_space()
        #action_seq.append(np.array([action]).astype(np.float32))
        action_seq.append(action)

    data = MountainCarData(num_sequences=1, sequence_length=len(action_seq), manual_action_seq=action_seq)
    model = GenerativeModel(state_dim=STATE_DIMENSION,
                            action_dim=ACTION_DIMENSION,
                            observation_dim=OBSERVATION_DIMENSION)
    model.load_state_dict("models")
    model.eval()
    model = model.to(DEVICE)

    positions = [obs[0] for obs in data.data[0][1]]
    positions_prior = []
    positions_posterior = []

    last_action_seq, cur_obs_seq = data.__getitem__(0)
    last_action_seq = last_action_seq.unsqueeze(0)#.to(DEVICE)
    cur_obs_seq = cur_obs_seq.unsqueeze(0)#.to(DEVICE)
    last_state = torch.tensor(np.zeros((1, STATE_DIMENSION)).astype(np.float32)).to(DEVICE)
    for t in range(len(last_action_seq[0])):
        last_action = last_action_seq[:, t]
        cur_observation = cur_obs_seq[:, t]
        if len(last_action.shape) == 1:
            last_action = last_action.unsqueeze(-1)
        if len(cur_observation.shape) == 1:
            cur_observation = cur_observation.unsqueeze(-1)

        SA_input = torch.hstack((last_state, last_action))
        SAO_input = torch.hstack((last_state, last_action, cur_observation))

        prior = model.prior(SA_input)
        prior_state = model.prior.sample_state(prior)

        posterior = model.posterior(SAO_input)
        posterior_state = model.posterior.sample_state(posterior)

        likelihood_posterior = model.likelihood(posterior_state)
        likely_observation_posterior = model.likelihood.sample_observation(likelihood_posterior)
        #likely_observation_posterior = model.likelihood.best_guess_observation(likelihood_posterior)
        positions_posterior.append(likely_observation_posterior[0, 0].item())

        likelihood_prior = model.likelihood(prior_state)
        likely_observation_prior = model.likelihood.sample_observation(likelihood_prior)
        #likely_observation_prior = model.likelihood.best_guess_observation(likelihood_prior)
        positions_prior.append(likely_observation_prior[0, 0].item())

    print(np.mean(positions), np.mean(positions_posterior), np.mean(positions) - np.mean(positions_posterior))
    print(torch.nn.MSELoss()(torch.tensor(np.array(positions).astype(np.float32)), torch.tensor(np.array(positions_posterior).astype(np.float32))))
    plt.plot(positions, label="Observed Pos")
    #plt.plot(positions_prior, label="Prior Sampled Pos")
    plt.plot(positions_posterior, label="Posterior Sampled Pos")
    plt.legend()
    plt.show()

    pass





