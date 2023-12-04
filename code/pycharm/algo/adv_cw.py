import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_adv_cw(
        model: nn.Module, adv_examples: torch.Tensor,
        adv_target: int = 3, iteration: int = 5000,
        lr: float = 0.01, c: float = 1
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_single_adv(
            model: nn.Module, adv_examples: torch.Tensor,
            adv_target: int = 3, iteration: int = 5000,
            lr: float = 0.01, c: float = 1
    ):
        box_max = 1
        box_min = 0
        box_mul = (box_max - box_min) / 2
        box_plus = (box_min + box_max) / 2
        modifier = torch.zeros_like(adv_examples, requires_grad=True)
        l2dist_list = []
        loss2_list = []
        loss_list = []
        model.eval()
        for i in range(iteration):
            new_example = torch.tanh(adv_examples + modifier) * box_mul + box_plus
            l2dist = torch.sum(torch.square(new_example - adv_examples))
            output = model(new_example)
            # 设定攻击目标
            onehot = torch.zeros_like(output)
            onehot[:, adv_target] = 1
            others = torch.max((1 - onehot) * output, dim=1).values
            real = torch.sum(output * onehot, dim=1)
            loss2 = torch.sum(torch.maximum(torch.zeros_like(others) - 0.01, others - real))
            loss = l2dist + c * loss2

            l2dist_list.append(l2dist)
            loss2_list.append(loss2)
            loss_list.append(loss)

            if modifier.grad is not None:
                modifier.grad.zero_()
            loss.backward()

            modifier = (modifier - modifier.grad * lr).detach()
            modifier.requires_grad = True

        def plot_loss(loss, loss_name):
            plt.figure(2, 2)
            plt.plot([i for i in range(len(loss))], [j.detach().numpy() for j in loss])
            # plt.yticks(np.arange(1, 50, 0.5))
            plt.xlabel('iteration times')
            plt.ylabel(loss_name)
            plt.show()

        plot_loss(l2dist_list, 'l2 distance loss')
        plot_loss(loss2_list, 'category loss')
        plot_loss(loss_list, 'all loss')
        new_img = torch.tanh(adv_examples + modifier) * box_mul + box_plus
        return new_img
    adv_list = []
    for i in adv_examples:
        adv_list.append(create_single_adv(model, i, adv_target, iteration, lr))
    return torch.Tensor(adv_list)
