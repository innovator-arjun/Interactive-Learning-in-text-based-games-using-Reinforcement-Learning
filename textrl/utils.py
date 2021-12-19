from torch.utils.data import Dataset
import torch
import numpy as np
import random
import os
import subprocess


def get_games(type, dir, level, game_count=100, seed=None):
    seed = random.randint(0, 1000000) if seed is None else seed
    games = []
    level_dir = os.path.join(dir, str(level))
    if not os.path.exists(level_dir):
        os.makedirs(level_dir)
    for file in os.listdir(level_dir):
        if file.endswith('.ulx'):
            games.append(level_dir + "/" + file)
    while len(games) < game_count:
        file = level_dir + "/" + type + str(len(games)) + "_" + level + ".ulx"
        command = "tw-make " + type + " --level " + level + " --output " + file
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
        games = []
        for file in os.listdir(level_dir):
            if file.endswith('.ulx'):
                games.append(file)
        print("Generated games:", len(games))

    random.seed(seed)
    random.shuffle(games)
    return games[:game_count]


def dqn_collate(batch):

    transposed = {
        'reward': [],
        'terminal': [],
        'state_prime': [],
        'commands_prime': [],
        'hidden_prime': [],
        'state': [],
        'commands': [],
        'actions': [],
        'hidden': []
    }

    for step in batch:
        transposed['reward'].append(step[0])
        transposed['terminal'].append(step[7])
        transposed['state_prime'].append(step[1])
        transposed['commands_prime'].append(step[2])
        transposed['hidden_prime'].append(step[3])
        transposed['state'].append(step[4])
        transposed['commands'].append(step[5])
        transposed['actions'].append(step[6])
        transposed['hidden'].append(step[8])

    transposed['reward'] = torch.FloatTensor(transposed['reward'])
    transposed['terminal'] = torch.FloatTensor(transposed['terminal'])
    transposed['actions'] = torch.LongTensor(transposed['actions'])

    return transposed


class ExperienceDataset(Dataset):
    def __init__(self, experience, keys):
        super(ExperienceDataset, self).__init__()
        self._keys = keys
        self._exp = []
        for x in experience:
            self._exp.append(x)

    def __getitem__(self, index):
        chosen_exp = self._exp[index]
        return tuple(chosen_exp[k] for k in self._keys)

    def __len__(self):
        return len(self._exp)


class PrioritizedReplay:
    def __init__(self, capacity, keys, epsilon=.01, alpha=.6):
        self._tree = SumTree(capacity)
        self.keys = keys
        self.e = epsilon
        self.a = alpha
        self.max_error = 5

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def extend(self, rollout):
        for step in rollout:
            self._tree.add(self.max_error, step)

    def sample(self, n):
        sample = []
        n = min(self._tree.length, n)
        segment = self._tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self._tree.get(s)
            data["idx"] = idx
            sample.append(data)
        return ExperienceDataset(sample, self.keys)

    def update(self, idx, error):
        self.max_error = max(self.max_error, error)
        p = self._get_priority(error)
        self._tree.update(idx, p)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.length = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
