import numpy as np
from copy import deepcopy
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import re

from textrl.utils import PrioritizedReplay, dqn_collate
from textrl.model import *


class TextAgent:
    def __init__(self, device="cpu", state_type="recurrent", train_freq=100, training_epochs=1,
                 batch_size=128, target_update=1000, sample_size=1000, lr=1e-3, eps=.99,
                 eps_decay=.9999, gamma=.9, max_vocab=1000, max_memory=30):
        self._initialized = False
        self._epsiode_has_started = False
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}
        self.device = device

        if state_type == "naive":
            self.model = NaiveNet(max_vocab, 128, device=device, use_last_action=True).to(device)
        elif state_type == "recurrent":
            self.model = RecurrentNet(max_vocab, 128, device=device, use_last_action=True).to(device)
       
        else:
            print("Invalid state type", state_type)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.value_objective = nn.functional.smooth_l1_loss
        self.target = deepcopy(self.model).to(device)

        self.mode = "test"
        self.transitions = []
        self.replay_buffer = PrioritizedReplay(1000000, ('reward', 'state_prime', 'commands_prime', 'hidden_prime',
                                                         'state', 'commands', 'actions', 'terminal', 'hidden'))
        self.no_train_step = 0
        self.train_freq = train_freq
        self.training_epochs = training_epochs
        self.target_update = target_update
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.eps = eps
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.max_vocab = max_vocab

    def train(self):
        self.mode = "train"
        self.transitions = []
        print(self.model)
        self.model.reset_hidden(1)
        self.no_train_step = 0

    def test(self):
        self.mode = "test"
        self.model.reset_hidden(1)

    def act(self, state, reward, done, admissible_commands):
        # Tokenize and pad the input and the commands to chose from.
        input_tensor = self._prepare_tensor([state])
        commands_tensor = self._prepare_tensor(admissible_commands)

        # Get our next action.
        outputs = self.model(input_tensor, commands_tensor)
        self.eps *= self.eps_decay
        if np.random.random() > self.eps:
            index = torch.argmax(outputs).unsqueeze(0)
        else:
            index = torch.randint(len(admissible_commands), (1,))
        self.model.last_action = index
        action = admissible_commands[index]

        # If testing finish
        if self.mode == "test":
            if done:
                self.model.reset_hidden(1)
            return action

        # Else record transitions
        self.no_train_step += 1

        # Train
        if self.no_train_step % self.train_freq == 0:
            dataset = self.replay_buffer.sample(self.sample_size)
            if len(dataset) > 0:
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False,
                                    collate_fn=dqn_collate)
                self._train(loader)
                self.model.reset_hidden(1)

        if self.no_train_step % self.target_update == 0:
            self.target.load_state_dict(self.model.state_dict())

        if self.transitions:
            self.transitions[-1]['reward'] = reward  # Update reward information.
            self.transitions[-1]['terminal'] = np.array([int(done)])  # Update terminal information.
            self.transitions[-1]['state_prime'] = input_tensor.cpu().detach().numpy()  # Update state information.
            self.transitions[-1]['commands_prime'] = commands_tensor.cpu().detach().numpy()  # Update action information.
            self.transitions[-1]['hidden_prime'] = self.model.get_hidden()  # Update hidden state information.

        if done:
            self.replay_buffer.extend(self.transitions)
            self.transitions = []
        else:
            # Keep information about transitions for Truncated Backpropagation Through Time.
            # Reward will be set on the next call
            self.transitions.append({'reward': None,  # Reward
                                     'terminal': None,  # Terminal
                                     'state_prime': None,  # Next State
                                     'commands_prime': None,  # Next Actions
                                     'hidden_prime': None,  # Next Hidden State
                                     'state': input_tensor.cpu().detach().numpy(),  # State
                                     'commands': commands_tensor.cpu().detach().numpy(),  # Actions
                                     'actions': index.cpu().detach().numpy(),  # Chosen Actions
                                     'hidden': self.model.get_hidden()})  # Hidden state of model
        return action

    def _train(self, loader):
        for epoch in range(self.training_epochs):
            for batch, step in enumerate(loader):
                self.optimizer.zero_grad()

                reward = step['reward'].to(self.device).unsqueeze(-1)
                terminal = step['terminal'].to(self.device)
                actions = step['actions'].to(self.device)

                model_q = self._get_model_out(self.model, step['state'], step['commands'], step['hidden'], indices=actions)
                target_out = self._get_model_out(self.target, step['state_prime'], step['commands_prime'], step['hidden_prime'])
                target_q = reward + self.gamma * target_out * (1 - terminal)

                loss = torch.mean(self.value_objective(target_q, model_q))
                loss.backward()
                self.optimizer.step()

    def _get_model_out(self, model, state, commands, hidden, indices=None):
        q_values = []
        for i in range(len(hidden)):
            s = torch.from_numpy(state[i]).to(self.device)
            c = torch.from_numpy(commands[i]).to(self.device)
            model.set_hidden(hidden[i])
            q = model(s, c)
            if indices is None:
                q = torch.max(q, 2)[0].squeeze(-1)
            else:
                q = q[0, 0, indices[i]]
            q_values.append(q)
        return torch.stack(q_values)

    def _get_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.max_vocab:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)

        return self.word2id[word]

    def _get_token(self, text):
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_id, text.split()))
        return word_ids

    def _prepare_tensor(self, texts):
        texts = list(map(self._get_token, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
        padded_tensor = padded_tensor.permute(1, 0)
        return padded_tensor
