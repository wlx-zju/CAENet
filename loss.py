import torch
import torch.nn as nn


def loss_calculation(pred, gt, task):
    if task == 'semantic':
        loss = nn.CrossEntropyLoss(ignore_index=255)(pred, gt)
    elif task == 'depth':
        mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(dim=1).to(pred.device)
        loss = torch.sum(torch.abs(pred-gt)*mask)/mask.sum().item()

    return loss


class LossSum(nn.Module):
    def __init__(self, tasks=['semantic', 'depth'], loss_weights=[1, 1]):
        super(LossSum, self).__init__()
        self.tasks = tasks
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        task_losses = {}
        for task in self.tasks:
            task_losses[task] = loss_calculation(pred[task], gt[task], task)

        loss = sum([self.loss_weights[i] * task_losses[task] for i, task in enumerate(self.tasks)])

        return loss, task_losses


def init_losses(tasks):
    losses = {}
    for task in tasks:
        losses[task] = 0
    return losses


def update_losses(losses, task_losses, batch_num):
    for task in task_losses:
        if task_losses[task] != 0:
            losses[task] += task_losses[task].item()/batch_num
