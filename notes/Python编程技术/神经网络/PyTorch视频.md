#### Tensor:

| python      | PyTorch     |
| ----------- | ----------- |
| Int         | IntTensor   |
| float       | FloatTensor |
| Int array   | IntTensor   |
| Float array | FloatTensor |
| string      | --          |

#### string:

One-hot: [0, 1, 0, 0]，存储起来信息密度太小，无法表示语言相关性

Embedding: Word2vec、glove



### 常用PyTorch方法

| 方法                                              | 作用                          |
| ------------------------------------------------- | ----------------------------- |
| torch.tensor()                                    | 创建tensor                    |
| a.type()                                          | 获得数据的类型                |
| a.numel()                                         | tensor占用内存的数量          |
| a.dim()                                           | 返回维数                      |
| Torch.empty()                                     | 生成非初始化数据              |
| Torch.FloatTensor(d1,d2,d3)                       | 生成对应维数的未初始化tensor  |
| torch.set_default_tensor_type(torch.DoubleTensor) | 设置默认生成的tensor数据类型  |
| torch.rand_like(a)                                | ==rand(a.shape)               |
| torch.randint(min, max, shape:[])                 | 生成随机整数tensor            |
| torch,randn()                                     | 生成标准正态分布              |
| torch.normal(mean=均值, std=方差)                 | 生成随机正态分布              |
| torch.full(shape:[], num)                         | 生成全部为某一个数值的tensor  |
| torch,arange(0, 10, 2)                            | 生成等差数列                  |
| torch.linespace(0, 10, steps=切分的分数)          | --                            |
| torch.logspace(0, -1, steps=底数)                 | 生成幂从0到-1，底数为10的数列 |
| torch.eye(shape)                                  | 生成单位矩阵                  |
| torch.randperm()                                  | 生成随机打散的tensor          |

#### 切片操作

|方法|作用|
|-------------------|---------------|
| a[0:28:2]                                         | 隔行索引，start:\end:step                                    |
| a.index_select(0, torch.tensor([0, 2]))           | 选择的哪一个维度，选择的范围                                 |
| a[0, ...]==a[0, :, :, :]                          | 代表任意长，根据实际形状推测，仅仅只是为了方便               |
| mask=x.ge(0.5); totch.masked_select(x, mask)      | 先获得大于等于0.5的数值在tensor中的位置，再打平成一维后选出tensor中大于等于0.5的数值 |

#### 视图信息

| 方法                  | 作用                                                         |
| --------------------- | ------------------------------------------------------------ |
| a.view(4, 1\*28\*28)  | 将所有的图片打平成一维，可用于全连接层。但是这个会丢失一部分数据。 |
| a.squeeze(index)      | 挤压，维度删减，给定删减的维度的位置                         |
| a.unsqueeze(index)    | 展开，插入一个新的维度的位置                                 |
| expand                | 维度扩展，并不增加数据，速度快，且不增加内存（推荐）         |
| repeat                | 维度扩展，复制值                                             |
| b.t()                 | 矩阵转置操作，只适用于二维的矩阵转置                         |
| a.transpose(id1, id2) | 只交换两个维度；[b, h, w, c]是numpy存储图片的格式，适用于高维的矩阵转置操作 |
| a.permute(0, 2, 1, 3) | 指定某维度的下标进行交换                                     |

##### Broadcasting：自动维度扩展

[32, 1, 1]或者[14, 14]之类的符合要扩展的dim=0|1的标准，如果dim>=2，则扩展会产生歧义，无法实现扩展。

从最后一个维度开始匹配。

#### 合并与切分

| 方法                                          | 作用                                                         |
| --------------------------------------------- | ------------------------------------------------------------ |
| torch.cat([tensor1, tensor2], dim=合并的维度) | 指定某几个tensor在某一维度（其它维度属性一样）进行合并，总维度数dim一样 |
| torch.stack([a1, a2], dim=插入新维度的位置)   | 会插入一个新的维度，相当于创建了一个新的分组，意味着两个的shape必须一致（与cat不同） |
| c.split(len\|[len1, len2], dim)               | 按照长度进行拆分（或指定具体的不同拆分长度，需保证加起来等于总共的长度），并指定进行拆分的维度dim；但是不能只拆分为一块（即只返回一个tensor） |
| c.chunk(num, dim)                             | 按照数量进行拆分                                             |

#### 数学运算

| 方法                        | 作用                                                         |
| --------------------------- | ------------------------------------------------------------ |
| torch.add或加号+            | --                                                           |
| torch.sub或减号-            | --                                                           |
| torch.mul或乘号*            | 矩阵点乘，逐元素相乘                                         |
| torch.div或除号/            | --                                                           |
| torch.mm()                  | 只适用于二维矩阵乘法                                         |
| torch.matmul()或者重载符号@ | 广泛的矩阵乘法（推荐），仍然只取最后的两维进行运算，其余的维度会进行broadcasting扩展 |
| a.pow(num)                  | 幂运算                                                       |
| a.sqrt()                    | 取平方根                                                     |
| a.rsqrt()                   | 取平方根的倒数                                               |
| torch.exp()                 | 取以e为底的幂运算                                            |
| torch.log()                 | 对数运算，默认以e为底                                        |
| a.floor()                   | 下取整                                                       |
| a.ceil()                    | 上取整                                                       |
| a.trunc()                   | 裁剪为整数                                                   |
| a.frac()                    | 裁剪为小数                                                   |
| a.round()                   | 四舍五入                                                     |
| a.clamp(min[, max])         | 将不在此区间的值拉入此区间的范围，拉成端点值                 |

#### 统计数据

| 方法                                 | 作用                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| a.grad.norm(2)                       | 求范数，打印梯度的模（2范数）                                |
| a.max()\|a.max(dim, keepdim: bool)   | 返回最大值；返回某一维度与原本shape一致的最大值张量          |
| a.min()\|a.min(dim, keepdim: bool)   | 返回最小值；返回某一维度与原本shape一致的最小值张量          |
| a.median()                           |                                                              |
| a.argmax()\|a.argmax(dim)            | 打平后求最大值索引；得到某一维度的                           |
| a.argmin()\|a.argmin(dim)            | 打平后求最小值索引；得到某一维度的                           |
| a.topk(k, dim, largest: bool)        | 求概率最大（最小）的前k个                                    |
| a.kthvalue(k, dim)                   | 求第k小的数值                                                |
| torch.where(cond>0.5, a, b)          | 当满足某一情境时取b中对应位置的值，不满足取a中对应的值，可以使用GPU高度并行化 |
| torch.gather(input, dim, index, out) | 将index中的索引（即label）与input中的元素关联起来，后可以直接根据索引进行查表操作 |

##### 比较大小

可以直接使用相关的运算比较符号，因为已经进行过重载了。

------------------------



### 一些知识点

#### Sigmoid

$$
\begin{aligned}
sigmoid(x) = \sigma &= \cfrac{1}{1+e^{-x}} \\
\sigma^{\prime} &= \sigma(1-\sigma) \\
\end{aligned}
$$

##### 梯度弥散

当使用Sigmoid函数做为激活函数时，由于在趋近于无穷的两端，导数趋近于0，即loss长时间得不到有效更新。

```python
torch.sigmoid(a)
```

#### Tanh

$$
\begin{aligned}
tanh(x) = \cfrac{e^x-e^{-x}}{e^x+e^{-x}} \\
\end{aligned}
$$

```
torch.tanh(a)
```

#### Rectified Linear Unit

$$
\begin{aligned}
f(x)=
\begin{cases} 
0,&x<0 \\
x,&x>0
\end{cases} \\

f(x)^{\prime}=
\begin{cases} 
0,&x<0 \\
1,&x>0
\end{cases}
\end{aligned}
$$

```
torch.relu(a)
```

#### 典型的loss损失函数

##### 均方差函数MSE（Mean Squared Error）

$$
\begin{aligned}
& loss = \sum[y-(xw+b)]^2 \\
& L2-norm = \lVert y-(xw+b) \rVert_2 = \sqrt{\sum(y-pred)^2} \\
& loss = norm\left(y-(xw+b)\right)^2
\end{aligned}
$$

```python
mse = F.mse_loss(y, pred)

w.requires_grad_()

torch.autograd.grad(loss, [w1, w2, ...(返回的偏微分)])  # 自动求导

mse.backward()  # 不会额外返回梯度信息，而是自动附着在每一个参数上
w1.grad
```



##### Cross Entropy loss：交叉熵

$$
\begin{aligned}
Entropy = -\sum_i{P(i)\log P(i)}
\end{aligned}
$$

**Entropy（熵）：**熵越高越稳定，没有惊喜度
$$
\begin{aligned}
H(p,q) &= -\sum{p(x)\log q(x)} \\
H(p,q) &= H(p)+D_{KL}(p|q) \qquad \text{---KL Divergence，即散度}\\
\end{aligned}
$$


**Cross Entropy（交叉熵）：**

+ p=q：cross Entropy = Entropy，即$D_{KL}=0$

+ for one-hot encoding：entropy = 1log1 = 0

**[0, 1]二分类的交叉熵形式：**

单个样本的交叉熵：$y$是标签类别，$\hat{y}$是预测值类别
$$
\begin{aligned}
L = -[y\log{\hat{y}}+(1-y)\log{(1-\hat{y})}]
\end{aligned}
$$
多个样本的交叉熵：
$$
\begin{aligned}
L = -\sum_{i=1}^N y_i\log{\hat{y}_i}+(1-y_i)\log{(1-\hat{y}_i)}
\end{aligned}
$$


##### Softmax

$$
\begin{aligned}
S(y_i) &= \cfrac{e^{y_i}}{\sum_j{e^{y_j}}} \\
\cfrac{\partial p_i}{\partial a_j} &= \begin{cases}p_i(1-p_j), &i=j \\ -p_j \cdot p_i, &i\neq j \end{cases} \\
\\
OR \\
\delta_{ij} &= \begin{cases} 1, &i=j \\ 0, &i \neq j \end{cases}\\
\cfrac{\partial p_i}{\partial a_j} &= p_i(\delta_{ij}-p_j)
\end{aligned}
$$

#### MLP多输出感知机

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\MLP多层感知机.png" alt="MLP多层感知机" style="zoom: 25%;" />

#### MLP多层感知机及反向传播

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\MLP-1.png" alt="MLP-1" style="zoom: 25%;" />

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\MLP-2.png" alt="MLP-2" style="zoom:25%;" />

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\MLP-3.png" alt="MLP-3" style="zoom:25%;" />

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\MLP-4.png" alt="MLP-4" style="zoom:25%;" />

### 神经网络基础概念及知识点

#### 多分类问题实战

##### 代码

**网络结构**

```python
w1, b1 = torch.randn(200, 784, requires_grad=True), \
         torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), \
         torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), \
         torch.zeros(10, requires_grad=True)
    
def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x
```

**训练**

```python
optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)

        logits = forward(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()
```

#### 全连接层

**nn.Linear**

```python
layer1 = nn.Linear(784, 200)
layer2 = nn.Linear(200, 200)
layer3 = nn.Linear(200, 10)

x = layer1(x)
x = layer2(x)
x = layer3(x)
```

**relu**

```python
x = layer1(x)
x = F.relu(x, inplace=True)

...
```

**Step.**

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
        	nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),# 封装好的
            # F.relu(x, inplace=True)  能够以更小的粒度进行管理
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),# LeakyReLU在负轴方向也有较小的值能够存在，避免了ReLU的梯度弥散现象
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )
     
     def forward(self, x):
        x = self.model(x)
        return x
```

#### GPU加速

```python
device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()# cuda()是老方法，不推荐
```

#### argmax

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\argmax.png" alt="argmax" style="zoom: 33%;" />

#### 测试实战

```python
test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.view(-1, 28*28)
    logits = forward(data)
    test_loss += criteon(logits, target).item()

	pred = logits.data.max(1)[1]
	correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	test_loss, correct, len(test_loader.dataset),
	100.*correct / len(test_loader.dataset)
```

#### Visdom可视化

**可视化工具安装，比较麻烦，只能接受numpy的数据：**

```python
pip install tensorboardX
```

**Visdom from Facebook：**

```python
# Step1. install
pip install visdom

# Step2. run server damon
python -m visdom.server
```

[Visdom官方网站](https://github.com/facebookresearch/visdom)

```python
from visdom import Visdom
viz = Visdom()

viz.line([0., 0.]: y, [0.]: x, win='小窗口的ID', [env='整个窗口的ID'], opts=dict(title='train loss'[, legend=['loss', 'acc']: 多条曲线标签]))
viz.line([loss.item()], [global_step], win='小窗口ID', update='append')

viz.images(data.view(-1, 1, 28, 28), win='x')# 可以直接使用tensor传入
viz.text(str(pred.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))
```

#### 过拟合和欠拟合

**泰勒展开本质思想就是函数拟合的思想，所以可以用多项式函数对任何一个函数进行拟合。**

#### 交叉验证

##### train-val-test：划分数据

```python
train_db = torchvision.datasets.MNIST('./mnist_data', train=True: 确定为训练集, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ]))

print('train', len(train_db), 'test:', len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
print('db1:', len(train_db), 'db2:', len(val_db))
train_loader = torch.utils.data.DataLoader(
	train_db,
	batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
	val_db,
	batch_size=batch_size, shuffle=True)
```

##### K交叉验证

#### Regularization（Weight Decay）：防止Overfitting

$$
\begin{aligned}
\text{L1-regularization} \\
J(\theta) &= -\cfrac{1}{N}\sum_{i=1}^N y_i\log{\hat{y}_i}+(1-y_i)\log{(1-\hat{y}_i)}+\lambda\sum_{i=1}^n \vert \theta_i \vert \\
&\vert \theta_i \vert \text{：这个是一范数} \\
&\lambda \text{：lambda是一个超参数} \\

\text{L2-regularization} \\
&J(W;X,y) + \cfrac{1}{2}\lambda \cdot \Vert W \Vert^2
\end{aligned}
$$

```python
# L2-regularization
# 只有Overfitting了才能使用weight_decay参数，否则将会导致模型准确率下降
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)

# L1-regularization
regularization_loss = 0
for param in model.parameters():
    regularization_loss += torch.sum(torch.abs(param))
classify_loss = criteon(logits, target)
loss = classify_loss + 0.01*regularization_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### 动量与学习率衰减

$$
\begin{aligned}
&\omega^{k+1} = \omega^{k}-\alpha\nabla f(\omega^{k}) \\
\\
&z^{k+1} = \beta z^k+\nabla f(\omega^k) \\
&\omega^{k+1} = \omega^k-\alpha z^{k+1} \\
&\text{$z^k$：代表上一次的更新方向}
\end{aligned}
$$

```python
# 1.
optimizer = optim.SGD(net.parameters(), args.lr,
                      momentum=args.momentum,# 设置动量
                      weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min')# 学习衰减

for epoch in xrange(args.start_epoch, args.epochs):
    train(train_loader, model, criterion, optimizer, epoch)
    result_avg, loss_val = validate(val_loader, model, criterion, epoch)
    scheduler.step(loss_val)
    
# 2.
# lr=0.05, epoch<30
# lr=0.005, 30<=epoch<60
# lr=0.0005, 60<=epoch<90
scheduler = StepLR(optimizer, step_size=30：一般为1K, gamma=0.1：倍数)
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)
```

#### Dropout

每一次batch_size训练的时候依概率掐灭一些连接，这样相当于降低了网络的复杂度，避免过拟合。

但是最终测试的时候是使用的全部的连接来判断。

```python
net_dropped = torch.nn.Sequential(
	torch.nn.Linear(784, 200),
    torch.nn.Dropout(0.5),# torch.nn.Dropout(p=dropout_prob)
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200)
)

for epoch in range(epochs):
    # train
    net_dropped.train()
    ...
    # test
    net_dropped.eval()
    ...
```

### 卷积神经网络

#### 卷积

**单方向的卷积操作：**
$$
\begin{aligned}
y(t) = x(t)\cdot h(t) = \int_{-\infty}^{+\infty}{x(\tau)h(t-\tau)d\tau}
\end{aligned}
$$
Input_channels：输入图片的通道数

Kernel_channels：卷积核数，即识别的模式数（或模糊，或边缘，或锐化）

Kernel_size：3x3

Stride：移动的步距

Padding：边缘补充的0

```python
layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
# out = layer.forward(x)# 一次前向计算
out = layer(x)# 会调用pytorch自己的__call__()函数

layer.weight# 打印信息

# 或者使用F中的conv2d()
out = F.conv2d(x, w, b, stride=1, padding=1)
```

#### 池化层与采样

