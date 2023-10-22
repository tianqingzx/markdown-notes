import unittest
import numpy as np


class TestYUVMethod(unittest.TestCase):
    def test_transpose(self):
        import torch
        ls = [
            [[1, 2],
             [3, 4],
             [5, 6]],
            [[7, 8],
             [9, 10],
             [11, 12]]
        ]   # (2, 3, 2)，参数从左往右，多维矩阵从外层到内层
        np_ls = np.array(ls)
        print(np_ls, np_ls.shape)
        np_ls = np.transpose(np_ls, (1, 2, 0))
        print(np_ls, np_ls.shape)
        tensor_ls = torch.tensor(np_ls)
        print(np_ls[0, 1])

    def test_bin(self):
        s_bin = bin(598 * 299 - 18)
        print(s_bin, "\n", s_bin[2:], "\n", type(s_bin), len(s_bin[2:]))

    def test_np_sub(self):
        ls_1 = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        ls_2 = np.array([
            [4, 5, 6],
            [7, 8, 9]
        ])
        print(ls_2 - ls_1)

    def test_random(self):
        import random
        random.seed(3)
        for i in range(10):
            print(random.random())

    def test_print_model(self):
        import torch
        from torchvision import models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet50(pretrained=True).to(device)
        model.eval()
        print(model)

    def test_random_normal(self):
        import numpy as np
        nls = np.random.normal(30.70, 5, [1, 10])
        print(nls)
        ls = [31.91, 34.09, 28.96, 33.60, 37.52, 34.99, 27.50, 24.98, 36.70, 25.57]
        print(np.mean(ls), np.var(ls, ddof=1), np.std(ls, ddof=1))
