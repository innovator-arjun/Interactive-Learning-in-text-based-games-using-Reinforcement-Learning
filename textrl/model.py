import torch
from torch import nn
from torch.nn import functional as F

class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu", use_last_action=False):
        super(RecurrentNet, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.obs_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.state_gru = nn.GRU(hidden_size, hidden_size)

        input_size = 2 * hidden_size
        self.dqn = nn.Sequential(nn.Linear(input_size, int(input_size * .75)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .75), int(input_size * .5)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .5), int(input_size * .25)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .25), 1))

        self.hidden_size = hidden_size
        self.state_hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.last_cmds = None
        self.last_action = None
        self.device = device
        self.use_last_action = use_last_action

    def forward(self, obs, commands):
        nb_cmds = commands.size(1)

        # Get state and command encodings
        if self.last_cmds is not None and self.use_last_action:
            obs = torch.cat((self.last_cmds[:, self.last_action], obs), dim=0)
        obs_embedding = self.embedding(obs)
        obs_encoder_hidden = self.obs_encoder_gru(obs_embedding)[1]
        cmds_embedding = self.embedding(commands)
        cmds_encoder_hidden = self.cmd_encoder_gru(cmds_embedding)[1]  # 1 x cmds x hidden

        # Get state representation
        state_hidden = self.state_gru(obs_encoder_hidden, self.state_hidden)[1]
        self.state_hidden = state_hidden

        # Get dqn input
        state_input = torch.stack([state_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden
        dqn_input = torch.cat((state_input, cmds_encoder_hidden.unsqueeze(0)), dim=-1)  # 1 x batch x cmds x hidden

        # Compute q
        scores = self.dqn(dqn_input).squeeze(-1)  # 1 x batch x cmds

        # Store last actions
        self.last_cmds = commands

        return scores

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        self.last_cmds = None
        self.last_action = None

    def get_hidden(self):
        c = self.last_cmds.detach().cpu().numpy() if self.last_cmds is not None else None
        return [self.state_hidden.detach().cpu().numpy(), c, self.last_action]

    def set_hidden(self, tensors):
        self.state_hidden = torch.from_numpy(tensors[0]).to(self.device)
        self.last_cmds = torch.from_numpy(tensors[1]).to(self.device) if tensors[1] is not None else None
        self.last_action = tensors[2]


class NaiveNet(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu", use_last_action=False):
        super(NaiveNet, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.obs_encoder_gru = nn.GRU(hidden_size, hidden_size)
        self.cmd_encoder_gru = nn.GRU(hidden_size, hidden_size)

        input_size = 2 * hidden_size
        self.dqn = nn.Sequential(nn.Linear(input_size, int(input_size * .75)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .75), int(input_size * .5)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .5), int(input_size * .25)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * .25), 1))

        self.hidden_size = hidden_size
        self.last_cmds = None
        self.last_action = None
        self.device = device
        self.use_last_action = use_last_action

    def forward(self, obs, commands):
        nb_cmds = commands.size(1)

        # Get state and command encodings
        if self.last_cmds is not None and self.use_last_action:
            obs = torch.cat((self.last_cmds[:, self.last_action], obs), dim=0)
        obs_embedding = self.embedding(obs)
        obs_encoder_hidden = self.obs_encoder_gru(obs_embedding)[1]
        cmds_embedding = self.embedding(commands)
        cmds_encoder_hidden = self.cmd_encoder_gru(cmds_embedding)[1]  # 1 x cmds x hidden

        # Get dqn input
        state_input = torch.stack([obs_encoder_hidden] * nb_cmds, 2)  # 1 x batch x cmds x hidden
        dqn_input = torch.cat((state_input, cmds_encoder_hidden.unsqueeze(0)), dim=-1)  # 1 x batch x cmds x hidden

        # Compute q
        scores = self.dqn(dqn_input).squeeze(-1)  # 1 x batch x cmds

        # Store last actions
        self.last_cmds = commands

        return scores

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        self.last_cmds = None
        self.last_action = None

    def get_hidden(self):
        c = self.last_cmds.detach().cpu().numpy() if self.last_cmds is not None else None
        return [self.state_hidden.detach().cpu().numpy(), c, self.last_action]

    def set_hidden(self, tensors):
        self.state_hidden = torch.from_numpy(tensors[0]).to(self.device)
        self.last_cmds = torch.from_numpy(tensors[1]).to(self.device) if tensors[1] is not None else None
        self.last_action = tensors[2]