import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_path = 'E:/ai_learning_resource/bikeshare/Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
rides.head()  # 输出部分数据
counts = rides['cnt'][:50]
# x = np.arange(len(counts))
# y = np.array(counts)
# plt.figure(figsize=(10, 7))
# plt.plot(x, y, 'o-')
# plt.xlabel('X')
# plt.ylabel('Y')

x = torch.tensor(np.arange(len(counts)))
y = torch.tensor(np.array(counts))
sz = 10  # 隐含层神经元的数量
weights = torch.randn([1, sz], requires_grad=True)  # 初始化输入层到隐含层的权重矩阵
biases = torch.randn(sz, requires_grad=True)  # 隐含层偏置
weights2 = torch.randn([sz, 1], requires_grad=True)  # 隐含层到输出层的权重矩阵

learning_rate = 0.001
losses = []
for i in range(2000):
    hidden = x.expand(sz, len(x)).t() * weights.expand(len(x), sz) + biases.expand(len(x), sz)

    hidden = torch.sigmoid(hidden)
    predictions = hidden.mm(weights2)

    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())

    if i % 100 == 0:
        # print('hidden', hidden)
        print('pred-y:', predictions - y)
        print('loss:', loss.t())
        print('[+] ', weights.grad.data)
        print('[-] ', biases.grad.data)

    loss.backward()
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)

    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
