import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, 
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should >= 1 !")
        elif num_layers == 1:
            self.is_linear = True
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.is_linear = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            if self.num_layers > 2:
                for _ in range(num_layers - 2):
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.is_linear:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class ProjectionHead(nn.Module):
    def __init__(self, 
        dim: int
    ) -> None:
        super(ProjectionHead, self).__init__()

        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        return x