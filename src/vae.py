import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.modules.loss as Loss

import numpy as np
import matplotlib.pyplot as plt

from src.data import MountainCarData, MountainCar
from src.constants import DEVICE

NUM_NEURONS = 20

EPOCHS = 100
BATCH_SIZE = 32

STATE_DIMENSION = 4
ACTION_DIMENSION = 1
OBSERVATION_DIMENSION = 2

#torch.autograd.set_detect_anomaly(True)

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_all(0)


class MultiNormalNegLogProbLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):

        normal = torch.distributions.MultivariateNormal()


class RecurrentVAE(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, observation_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim+self.action_dim+self.observation_dim, NUM_NEURONS),
            nn.ReLU(),
            nn.Linear(NUM_NEURONS, NUM_NEURONS),
            nn.ReLU(),
            nn.Linear(NUM_NEURONS, 2*self.state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.state_dim, NUM_NEURONS),
            nn.ReLU(),
            nn.Linear(NUM_NEURONS, NUM_NEURONS),
            nn.ReLU(),
            nn.Linear(NUM_NEURONS, 2*self.observation_dim)
        )

    def forward(self, x: torch.tensor):
        state_distribution = self.encoder(x).reshape((-1, 2, self.state_dim))
        state_mu = nn.LeakyReLU()(state_distribution[:, 0])
        state_std = nn.ReLU()(state_distribution[:, 1])
        state_sample = self.sample_normal(state_mu, state_std)

        observation_distribution = self.decoder(state_sample).reshape((-1, 2, self.observation_dim))
        obs_mu = nn.LeakyReLU()(observation_distribution[:, 0])
        obs_std = nn.ReLU()(observation_distribution[:, 1])
        observation_sample = self.sample_normal(obs_mu, obs_std)

        return state_sample, observation_sample

    def sample_normal(self, mu: torch.tensor, std: torch.tensor):
        std_normal_noise = np.random.standard_normal(np.prod(mu.shape)).reshape(mu.shape)
        return mu + torch.tensor(std_normal_noise.astype(np.float32)).to(DEVICE) * std


def train(model: RecurrentVAE, dataloader: DataLoader,
          criterion: Loss, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    mse_acc = []
    for batch, (last_action_seq, cur_obs_seq) in enumerate(dataloader):
        last_action_seq, cur_obs_seq = last_action_seq.to(DEVICE), cur_obs_seq.to(DEVICE)

        last_state = torch.tensor(np.zeros((BATCH_SIZE, STATE_DIMENSION)).astype(np.float32)).to(DEVICE)
        loss = None
        for t in range(len(last_action_seq[0])):
            last_action = last_action_seq[:, t]
            cur_obs = cur_obs_seq[:, t]
            if len(last_action.shape) == 1:
                last_action = last_action.unsqueeze(-1)
            if len(cur_obs.shape) == 1:
                cur_obs = cur_obs.unsqueeze(-1)

            SAO = torch.hstack((last_state, last_action, cur_obs))

            sampled_state, sampled_obs = model(SAO)

            # Only train for position reconstruction
            cur_obs = cur_obs[:, 0]
            sampled_obs = sampled_obs[:, 0]

            if loss is None:
                loss = criterion(cur_obs, sampled_obs)
            else:
                loss += criterion(cur_obs, sampled_obs)

            last_state = sampled_state

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"Batch [{batch}], {(batch + 1) / len(dataloader):.2%} LR: [{optimizer.param_groups[0]['lr']:.6f}], MSE: [{loss.item():.6f}]")

        mse_acc.append(loss.item())
    return np.mean(mse_acc)


def eval(model: RecurrentVAE, dataloader: DataLoader, criterion: Loss) -> float:
    model.eval()
    mse_acc = []
    for batch, (last_action_seq, cur_obs_seq) in enumerate(dataloader):
        last_action_seq, cur_obs_seq = last_action_seq.to(DEVICE), cur_obs_seq.to(DEVICE)

        last_state = torch.tensor(np.zeros((BATCH_SIZE, STATE_DIMENSION)).astype(np.float32)).to(DEVICE)
        loss = None
        for t in range(len(last_action_seq[0])):
            last_action = last_action_seq[:, t]
            cur_obs = cur_obs_seq[:, t]

            if len(last_action.shape) == 1:
                last_action = last_action.unsqueeze(-1)
            if len(cur_obs.shape) == 1:
                cur_obs = cur_obs.unsqueeze(-1)

            SAO = torch.hstack((last_state, last_action, cur_obs))

            sampled_state, sampled_obs = model(SAO)

            # Only test for position reconstruction
            cur_obs = cur_obs[:, 0]
            sampled_obs = sampled_obs[:, 0]

            if loss is None:
                loss = criterion(cur_obs, sampled_obs)
            else:
                loss += criterion(cur_obs, sampled_obs)

            last_state = sampled_state

        if batch % 10 == 0:
            print(f"Eval Batch [{batch}], {(batch + 1) / len(dataloader):.2%}, MSE: [{loss.item():.6f}]")

        mse_acc.append(loss.item())
    return np.mean(mse_acc)


def test(model: RecurrentVAE, criterion: Loss):
    action_seq = []
    for i in range(300):
        action = MountainCar(step_lim=np.inf).sample_action_space()
        action_seq.append(action)
    data = MountainCarData(num_sequences=1, sequence_length=len(action_seq), manual_action_seq=action_seq)

    positions = [obs[0] for obs in data.data[0][1]]
    positions_posterior = []

    last_action_seq, cur_obs_seq = data.__getitem__(0)
    last_action_seq = last_action_seq.unsqueeze(0)  # .to(DEVICE)
    cur_obs_seq = cur_obs_seq.unsqueeze(0)  # .to(DEVICE)

    last_state = torch.tensor(np.zeros((1, STATE_DIMENSION)).astype(np.float32)).to(DEVICE)
    loss = None
    for t in range(len(last_action_seq[0])):
        last_action = last_action_seq[:, t]
        cur_obs = cur_obs_seq[:, t]

        if len(last_action.shape) == 1:
            last_action = last_action.unsqueeze(-1)
        if len(cur_obs.shape) == 1:
            cur_obs = cur_obs.unsqueeze(-1)

        SAO = torch.hstack((last_state, last_action, cur_obs))

        sampled_state, sampled_obs = model(SAO)

        # Only test for position reconstruction
        cur_obs = cur_obs[:, 0]
        sampled_obs = sampled_obs[:, 0]

        if loss is None:
            loss = criterion(cur_obs, sampled_obs)
        else:
            loss += criterion(cur_obs, sampled_obs)

        last_state = sampled_state

        positions_posterior.append(sampled_obs[0].item())

    print("Loss: ", loss)
    plt.plot(positions, label="Observed Pos")
    plt.plot(positions_posterior, label="Posterior Sampled Pos")
    plt.legend()
    plt.savefig('models/vae_out.png')


def main():
    training_data = MountainCarData(num_sequences=10_000, sequence_length=300, load_path="data/train_10k_300")
    eval_data = MountainCarData(num_sequences=10_000, sequence_length=300, load_path="data/eval_10k_300")

    train_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = RecurrentVAE(state_dim=STATE_DIMENSION, action_dim=ACTION_DIMENSION, observation_dim=OBSERVATION_DIMENSION)
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=0.1)
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    model.load_state_dict(torch.load('models/vae.pth'))
    test(model, criterion)
    return

    avg_mse_train = []
    avg_mse_eval = []
    best_eval = np.inf
    for epoch in range(EPOCHS):
        epoch_mse_train = train(model, train_loader, criterion, optimizer)
        epoch_mse_eval = eval(model, eval_loader, criterion)

        avg_mse_train.append(epoch_mse_train)
        avg_mse_eval.append(epoch_mse_eval)

        if epoch_mse_eval < best_eval:
            best_eval = epoch_mse_eval
            torch.save(model.state_dict(), 'models/vae.pth')

        scheduler.step(epoch_mse_eval)

        print(f"\nEpoch [{epoch}] complete, Avg MSE (train) [{epoch_mse_train:.4f}, Avg MSE (dev) [{epoch_mse_eval:.4f}]\n")

    print("Done.")


if __name__ == "__main__":
    main()
