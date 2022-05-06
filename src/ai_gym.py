import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader

from src.model import GenerativeModel
from src.data import MountainCarData, RandomMountainCarData
from src.constants import DEVICE

torch.manual_seed(0)
np.random.seed(0)

#torch.autograd.set_detect_anomaly(True)


STATE_DIMENSION = 4
ACTION_DIMENSION = 1
OBSERVATION_DIMENSION = 2

BATCH_SIZE = 1
EPOCHS = 50


def train(model, dataloader, optimizer):
    model.train()
    free_energy = []
    for batch, (last_action_seq, cur_obs_seq) in enumerate(dataloader):
        last_action_seq, cur_obs_seq = last_action_seq.to(DEVICE), cur_obs_seq.to(DEVICE)

        last_state = torch.tensor(np.zeros((BATCH_SIZE, STATE_DIMENSION)).astype(np.float32)).to(DEVICE)
        kl_div = 0.0
        log_likelihood_mse = 0.0

        for t in range(len(last_action_seq[0])):
            last_action = last_action_seq[:, t]
            cur_observation = cur_obs_seq[:, t]
            if len(last_action.shape) == 1:
                last_action = last_action.unsqueeze(-1)
            if len(cur_observation.shape) == 1:
                cur_observation = cur_observation.unsqueeze(-1)

            # SA_input = torch.hstack((last_state, last_action))
            SAO_input = torch.hstack((last_state, last_action, cur_observation))

            # prior = model.prior(SA_input)
            # prior_state = model.prior.sample_state(prior)

            posterior = model.posterior(SAO_input)
            posterior_state = model.posterior.sample_state(posterior)

            likelihood = model.likelihood(posterior_state)
            likely_observation = model.likelihood.sample_observation(likelihood)

            if t > 0:
                # Q = torch.distributions.Normal(posterior[:, 0].squeeze(), posterior[:, 1].squeeze())
                # P = torch.distributions.Normal(prior[:, 0].squeeze(), prior[:, 1].squeeze())
                # kl_div += torch.distributions.kl_divergence(P, Q)

                # O = torch.distributions.Normal(likelihood[:, 0].squeeze(), likelihood[:, 1].squeeze())
                mse_loss = torch.nn.MSELoss()
                # log_likelihood_mse += -mse_loss(likely_observation[:, 0].squeeze(), cur_observation[:, 0].squeeze())
                recon_errors = likely_observation[:, 0].squeeze() - cur_observation[:, 0].squeeze()
                log_likelihood_mse -= torch.mean(torch.square(recon_errors))
                # log_likelihood += O.log_prob(cur_observation)

            last_state = posterior_state.detach()

        # kl_div = kl_div.mean()
        # log_likelihood = log_likelihood[:, 0].mean()

        # F = kl_div - log_likelihood_mse
        F = -log_likelihood_mse
        optimizer.zero_grad()
        F.backward()
        optimizer.step()



        if batch % 10 == 0:
            print(f"Batch [{batch}], {(batch + 1) / len(dataloader):.2%} Free Energy: [{F.item():>7f}]"
                  f" LR: [{optimizer.param_groups[0]['lr']:.6f}], KL: [{kl_div if isinstance(kl_div, float) else kl_div.item():.6f}], MSE: [{-log_likelihood_mse.item():.6f}]")

        free_energy.append(F.item())

    return np.mean(free_energy)


def test(model, dataloader):
    model.eval()
    free_energy = []
    for batch, (last_action_seq, cur_obs_seq) in enumerate(dataloader):
        last_action_seq, cur_obs_seq = last_action_seq.to(DEVICE), cur_obs_seq.to(DEVICE)

        last_state = torch.tensor(np.zeros((BATCH_SIZE, STATE_DIMENSION)).astype(np.float32)).to(DEVICE)
        kl_div = 0.0
        log_likelihood_mse = 0.0

        for t in range(len(last_action_seq[0])):
            last_action = last_action_seq[:, t]
            cur_observation = cur_obs_seq[:, t]
            if len(last_action.shape) == 1:
                last_action = last_action.unsqueeze(-1)
            if len(cur_observation.shape) == 1:
                cur_observation = cur_observation.unsqueeze(-1)

            # SA_input = torch.hstack((last_state, last_action))
            SAO_input = torch.hstack((last_state, last_action, cur_observation))

            # prior = model.prior(SA_input)
            # prior_state = model.prior.sample_state(prior)

            posterior = model.posterior(SAO_input)
            posterior_state = model.posterior.sample_state(posterior)

            likelihood = model.likelihood(posterior_state)
            likely_observation = model.likelihood.sample_observation(likelihood)

            if t > 0:
                # Q = torch.distributions.Normal(posterior[:, 0].squeeze(), posterior[:, 1].squeeze())
                # P = torch.distributions.Normal(prior[:, 0].squeeze(), prior[:, 1].squeeze())
                # kl_div += torch.distributions.kl_divergence(P, Q)

                # O = torch.distributions.Normal(likelihood[:, 0].squeeze(), likelihood[:, 1].squeeze())
                mse_loss = torch.nn.MSELoss()
                # log_likelihood_mse += -mse_loss(likely_observation[:, 0].squeeze(), cur_observation[:, 0].squeeze())
                recon_errors = likely_observation[:, 0].squeeze() - cur_observation[:, 0].squeeze()
                # recon_errors[recon_errors==0] = 1e-12
                log_likelihood_mse -= torch.mean(torch.square(recon_errors))
                # log_likelihood += O.log_prob(cur_observation)
            last_state = posterior_state

        # kl_div = kl_div.mean()
        # log_likelihood = log_likelihood[:, 0].mean()

        # F = kl_div - log_likelihood_mse
        F = -log_likelihood_mse

        if batch % 10 == 0:
            print(f"Eval Batch [{batch}], {(batch + 1) / len(dataloader):.2%} Free Energy: [{F.item():>7f}] "
                  f"KL: [{kl_div if isinstance(kl_div, float) else kl_div.item():.6f}], MSE: [{-log_likelihood_mse.item():.6f}]")

        free_energy.append(F.item())

    return np.mean(free_energy)


if __name__ == "__main__":
    training_data = MountainCarData(num_sequences=10_000, sequence_length=300, load_path="data/train_10k_300")
    eval_data = MountainCarData(num_sequences=10_000, sequence_length=300, load_path="data/eval_10k_300")

    #training_data = RandomMountainCarData(num_sequences=10_000, sequence_length=300)  # , load_path="data")
    #eval_data = MountainCarData(num_sequences=10_000, sequence_length=300)#, load_path="data")

    train_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = GenerativeModel(state_dim=STATE_DIMENSION,
                            action_dim=ACTION_DIMENSION,
                            observation_dim=OBSERVATION_DIMENSION).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, threshold=0.1)
    for m in model.models:
        for p in m.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    avg_free_energy_train= []
    avg_free_energy_test = []
    best_free_energy_test = np.inf
    for epoch in range(EPOCHS):
        epoch_free_energy_train = train(model, train_loader, optimizer)

        if np.isnan(epoch_free_energy_train):
            break
        avg_free_energy_train.append(epoch_free_energy_train)

        epoch_free_energy_test = test(model, eval_loader)

        if np.isnan(epoch_free_energy_test):
            break
        avg_free_energy_test.append(epoch_free_energy_test)
        scheduler.step(avg_free_energy_test[-1])
        if avg_free_energy_test[-1] < best_free_energy_test:
            best_free_energy = avg_free_energy_test[-1]
            model.save_state_dict("models")

        print(f"\nEpoch [{epoch}] complete, Avg Free Energy (train) [{epoch_free_energy_train:.4f}], Avg Free Energy (dev) [{epoch_free_energy_test:.4f}]\n")

    me = 1 + 1

