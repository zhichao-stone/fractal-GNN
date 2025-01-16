import torch
import torch.nn.functional as F


### controstive learning
class ContrastiveLearningLoss:
    def __init__(self, temperature: float = 0.5) -> None:
        self.temperature = temperature

    def __call__(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class CLLoss1(ContrastiveLearningLoss):
    def __init__(self, temperature: float = 0.5) -> None:
        super(CLLoss1, self).__init__(temperature)

    def __call__(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        sim_matrix = torch.matmul(F.normalize(v1), F.normalize(v2).T) / self.temperature

        row_softmax_matrix = F.log_softmax(sim_matrix, dim=1)
        col_softmax_matrix = F.log_softmax(sim_matrix, dim=0)

        row_diag_sum = row_softmax_matrix.diag().sum(-1)
        col_diag_sum = col_softmax_matrix.diag().sum(-1)
        contrastive_loss = - (row_diag_sum + col_diag_sum) / (2*len(row_softmax_matrix))

        return contrastive_loss

class CLLoss2(ContrastiveLearningLoss):
    def __init__(self, temperature: float = 0.5) -> None:
        super(CLLoss2, self).__init__(temperature)

    def __call__(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        sim_matrix = torch.matmul(F.normalize(v1), F.normalize(v2).T) / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        pos_sim = sim_matrix.diag()
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        contrastive_loss = - torch.log(loss).mean()

        return contrastive_loss

class CLLoss3(ContrastiveLearningLoss):
    def __init__(self, temperature: float = 0.5) -> None:
        super(CLLoss3, self).__init__(temperature)

    def __call__(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        sim_matrix = torch.matmul(F.normalize(v1), F.normalize(v2).T) / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        pos_sim = sim_matrix.diag()
        row_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        col_loss = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)

        contrastive_loss = - (torch.log(row_loss) + torch.log(col_loss)) / (sim_matrix.size(0))

        return contrastive_loss
    