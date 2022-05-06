import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.stats
from torchvision.transforms import ToTensor
import gym
import pickle

from src.constants import DEVICE

class NormalDistribution(Dataset):
    def __init__(self, num_samples: int, mean: float = 0, std: float = 1):
        self.data = np.random.normal(mean, std, size=num_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value = self.data[idx]
        prob = scipy.stats.norm.pdf(1, loc=0, scale=1)
        return value, prob


class MountainCar:
    def __init__(self, step_lim: int):
        self.done = False
        self.step_lim = step_lim
        self.cur_step = 0
        self.reward = 0
        self.env = gym.make('MountainCarContinuous-v0')

        self.current_observation = self.env.reset()
        self.last_action = None

    def step(self, action: float) -> bool:
        assert -1 <= action <= 1, f"MountainCar actions are numbers [-1, 1]. Got: {action}"

        self.current_observation, reward, done, _ = self.env.step(action)
        self.cur_step += 1
        self.reward += reward if not self.done else 0
        self.last_action = action
        self.done = done if not self.done else self.done
        return True

    def sample_action_space(self) -> float:
        return self.env.action_space.sample()


class MountainCarData(Dataset):
    def __init__(self, num_sequences: int, sequence_length: int = 100, choice_volatility: float = 0.05,
                 load_path: str = None, manual_action_seq: list = None):
        super().__init__()
        sequence_length -= 1
        self.data = []
        if load_path is None:
            print("Generating action/observation sequences...")
            last_perc = 0
            for i in range(num_sequences):
                cur_perc = (((i+1) / num_sequences) * 10) // 1
                if cur_perc != last_perc:
                    last_perc = cur_perc
                    print(f"{(i+1)/num_sequences:.1%}")
                sim = MountainCar(step_lim=sequence_length)
                action_seq = np.array([])
                obs_seq = np.array([])
                a = sim.sample_action_space()
                i = 0
                while sim.cur_step < sim.step_lim:
                    if manual_action_seq is not None:
                        a = manual_action_seq[i]
                    else:
                        if np.random.uniform(0, 1) <= choice_volatility:
                            a = sim.sample_action_space()
                    action_seq = np.append(a, action_seq)
                    sim.step(a)
                    if len(obs_seq) == 0:
                        obs_seq = sim.current_observation
                    else:
                        obs_seq = np.vstack((obs_seq, sim.current_observation))
                    i += 1
                self.data.append((action_seq, obs_seq))
                sim.env.close()
        else:
            self.load(load_path)

        self.actions = torch.tensor(np.array([d[0] for d in self.data]).astype(np.float32)).to(DEVICE)
        self.observations = torch.tensor(np.array([d[1] for d in self.data]).astype(np.float32)).to(DEVICE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> (torch.tensor, torch.tensor):
        #action_seq, obs_seq = self.data[idx]
        #action_seq = torch.tensor(action_seq.astype(np.float32))
        #obs_seq = torch.tensor(obs_seq.astype(np.float32))
        action_seq = self.actions[idx]
        obs_seq = self.observations[idx]
        return action_seq, obs_seq

    def save(self, path: str):
        with open(path, 'wb') as fp:
            pickle.dump(self.data, fp)

    def load(self, path: str):
        with open(path, 'rb') as fp:
            self.data = pickle.load(fp)


class RandomMountainCarData(Dataset):
    def __init__(self, num_sequences: int, sequence_length: int = 100, choice_volatility: float = 0.05):
        super().__init__()
        self.num_sequences = num_sequences
        self.sequence_len = sequence_length
        self.vol = choice_volatility

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx) -> (torch.tensor, torch.tensor):
        sim = MountainCar(step_lim=self.sequence_len)
        action_seq = np.array([])
        obs_seq = np.array([])
        a = sim.sample_action_space()
        i = 0
        while sim.cur_step < sim.step_lim:
            if np.random.uniform(0, 1) <= self.vol:
                a = sim.sample_action_space()
            action_seq = np.append(a, action_seq)
            sim.step(a)
            if len(obs_seq) == 0:
                obs_seq = sim.current_observation
            else:
                obs_seq = np.vstack((obs_seq, sim.current_observation))
            i += 1
        sim.env.close()

        action_seq = torch.tensor(action_seq.astype(np.float32))
        obs_seq = torch.tensor(obs_seq.astype(np.float32))
        return action_seq, obs_seq

    def save(self, path: str):
        with open(path, 'wb') as fp:
            pickle.dump(self.data, fp)

    def load(self, path: str):
        with open(path, 'rb') as fp:
            self.data = pickle.load(fp)
