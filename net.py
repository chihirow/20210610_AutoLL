import numpy as np
import torch
import torch.nn as nn


class AutoLL(nn.Module):  # Automatic linear layout network
    def __init__(self, n_units_in, n_units_out):
        super(AutoLL, self).__init__()
        self.layers_in = nn.ModuleList()  # Input layers
        for i in range(n_units_in.shape[0] - 1):
            self.layers_in.append(nn.Linear(n_units_in[i], n_units_in[i + 1]))
            self.layers_in.append(nn.Sigmoid())
        for i in np.arange(0, 2 * (n_units_in.shape[0] - 1), 2):
            interval_u = 1 / np.sqrt(n_units_in[i // 2])
            nn.init.uniform_(self.layers_in[i].weight, a=-interval_u, b=interval_u)
            nn.init.zeros_(self.layers_in[i].bias)

        self.layers_out = nn.ModuleList()  # Output layers
        for i in range(n_units_out.shape[0] - 1):
            self.layers_out.append(nn.Linear(n_units_out[i], n_units_out[i + 1]))
            self.layers_out.append(nn.Sigmoid())
        for i in np.arange(0, 2 * (n_units_out.shape[0] - 1), 2):
            interval_u = 1 / np.sqrt(n_units_out[i // 2])
            nn.init.uniform_(self.layers_out[i].weight, a=-interval_u, b=interval_u)
            nn.init.zeros_(self.layers_out[i].bias)

    def forward(self, x_row, x_col):
        for layers_in in self.layers_in:
            x_row = layers_in(x_row)
            x_col = layers_in(x_col)

        y = torch.cat((x_row, x_col), 1)  # Concatenate the row and column outputs
        for layers_out in self.layers_out:
            y = layers_out(y)

        return y, x_row, x_col


class Loss:
    def __init__(self, net):
        self.net = net

    def calc_loss(self, x_row, x_col, y, lambda_reg):
        y_model, _, _ = self.net(x_row, x_col)
        criterion = torch.nn.BCELoss()  # torch.nn.MSELoss()
        loss = criterion(y_model, y)

        device = x_row.device
        reg = torch.tensor(0.).to(device, dtype=torch.float)
        for param in self.net.parameters():
            reg += torch.norm(param) ** 2
        loss += lambda_reg * reg

        return loss
