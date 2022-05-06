import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from src.constants import DEVICE
from src.data import NormalGammaObservableDistribution
from src.loss import Surprise
from src.model import NeuralNetwork

training_data = NormalGammaObservableDistribution(num_samples=10000,
                                                  latent_shape=2,
                                                  latent_scale=1, observation_noise=1)

batch_size = 256

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


model = NeuralNetwork().to(DEVICE)
print(model)

loss_fn = Surprise()#nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

"""
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (observation, latent) in enumerate(dataloader):
        observation, latent = observation.to(DEVICE), latent.to(DEVICE)

        # Compute prediction error
        pred = model(observation)
        batch_mean = torch.mean(observation)
        batch_std = torch.std(observation) * torch.sqrt(observation)
        pred_mean = pred
        loss = torch.mean(torch.square(-pred - latent))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(observation)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
"""

def get_normal_pdf(mean: float, std: float):
    def pdf(x):
        return (1 / (std * np.sqrt(2*np.pi))) * np.power(np.e, -0.5 * np.power(((x - mean)/(std)), 2))
    return pdf

def train(model, optimizer, dataset: NormalGammaObservableDistribution):
    mu, std = 3, 1

    observations = dataset.observable_data
    data = [[np.random.uniform(low=-10, high=10) for _ in range(256)] for _ in range(1000)]
    pdf = get_normal_pdf(mu, std)
    for i, sample in enumerate(data):
        density_vals = [pdf(s) for s in sample]
        sample = torch.tensor(np.array(sample).reshape(-1, 1).astype(np.float32)).to(DEVICE)
        densities = torch.tensor(np.array(density_vals).reshape(-1, 1).astype(np.float32)).to(DEVICE)
        pred_densities = model(sample)
        loss = torch.mean(torch.square(pred_densities - densities))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i * len(data)
            print(f"loss: {loss:>7f} [{current:>5d}/{len(data):>5d}]")



def test(dataloader: DataLoader, model):
    latent = dataloader.dataset.latent_data
    #observed =


fig, axis = plt.subplots(2,2)

inputs = np.array([[i/100] for i in range(-1000, 1000, 1)])
model.eval()
outputs = model(torch.tensor(inputs).float().to(DEVICE))
axis[1,0].scatter(inputs, outputs.detach().cpu().numpy())
axis[1,0].set_title(f"Prior, mean: {np.mean(outputs.detach().cpu().numpy()):.2f} std: {np.std(outputs.detach().cpu().numpy()):.2f}")

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, optimizer, training_data)

axis[0,0].hist(training_data.latent_data, bins=100)
axis[0,0].set_title(f"Latent Samples, mean: {np.mean(training_data.latent_data):.2f}, std: {np.std(training_data.latent_data):.2f}")
axis[0,1].hist(training_data.observable_data, bins=100)
axis[0,1].set_title(f"Observable Samples, mean: {np.mean(training_data.observable_data):.2f}, std: {np.std(training_data.observable_data):.2f}")

inputs = np.array([[i/100] for i in range(-5000, 5000, 1)])
model.eval()
outputs = model(torch.tensor(inputs).float().to(DEVICE))
axis[1,1].scatter(inputs, outputs.detach().cpu().numpy())
axis[1,1].set_title(f"Posterior, mean: {np.mean(outputs.detach().cpu().numpy()):.2f} std: {np.std(outputs.detach().cpu().numpy()):.2f}")
plt.show()
print("Done!")


if __name__ == '__main__':
    pass
