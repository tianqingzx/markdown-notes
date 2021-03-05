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

#### 时间序列表示方法

**Batch:**

+ [word num, b, word vec]
+ [b, word num, word vec]

**word2vec查表操作**

```python
word_to_ix = {"hello": 0, "world": 1}
lookup_tensor = torch.tensor([word_to_ix["helo"]], dtype=torch.long)

embeds = nn.Embedding(2, 5)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
```

**CloVe查表操作**

```python
from torchnlp.word_to_vector import GloVe
vectors = GloVe()# 大约2.2G

vectors['hello']
```



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

Max pooling

Avg pooling

```python
layer = nn.MaxPool2d(2:窗口大小, stride=2)

out = F.avg_pool2d(x: 输入, 2, stride=2)
```

upsample：放大图片，就近取值放大

```python
out = F.interpolate(x, scale_factor=2: 放大的倍数, mode='nearest')
```

ReLU

```python
layer = nn.ReLU(inplace=True: x的导数会覆盖掉原本x的值)

out = F.relu(x)
```

#### Feature scaling

##### Image Normalization

```python
normalize = transforms.Normalize(mean=[0.485: R, 0.456: G, 0.406: B],
                                std=[0.229, 0.224, 0.225])
```

会将RGB三个通道全部转化为符合标准正态分布的区间中。

代码中的数据是统计得出的平均值。

##### Batch Normalization

例如对于Batch Norm来说：[6, 3, 28*28]：[N, C, H\*W]，表示六张三通道的图片，处理后生成[3]个数据做为每一个通道统计的平均值

用于避免或减轻梯度弥散现象

```python
x = torch.rand(100, 16, 784)
layer = nn.BatchNorm1d(16)
out = layer(x)

layer.running_mean# 生成的全局均值
layer.running_var# 生成的全局方差

vars(layer)# 打印全局一些参数
```

$$
\begin{aligned}
\mu &\gets \cfrac{1}{m}\sum_{i=1}^{m}{x_i} \qquad &\text{mini-batch mean} \\
\sigma^2 &\gets \cfrac{1}{m}\sum_{i=1}^{m}{(x_i-\mu)^2} \qquad &\text{mini-batch variance} \\
\hat{x_i} &\gets \cfrac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} \qquad &\text{normalize} \\
y_i &\gets \gamma \cdot \hat{x_i} + \beta \sim N(\beta, \gamma) \qquad &\text{scale and shift} \\
\end{aligned}
$$

#### 深度残差网络

使用一个短路连接，使得网络自己训练寻找最优的网络层数和结构

```python
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
            	nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )
            
     def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out
```

DenseNet：每一层跟前面的每一层直接相连，所以每一层都是综合了前面一些层数的信息

#### nn.Module

**save and load**

```python
device = torch.device('cuda')
net = Net()
net.to(device)

net.load_state_dict(torch.load('ckpt.md1'))

# train

torch.save(net.state_dict(), 'ckpt.md1')
```

**train/test**

```python
device = torch.device('cuda')
net = Net()
net.to(device)

# train
net.train()

# test
net.eval()
...
```

**implement own layer：实现展平操作**

```python
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class TestNet():
    def __init__(self):
        super(TestNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, stride=1, padding=1),
                                nn.MaxPool2d(2, 2),
                                Flatten(),
                                nn.Linear(1*14*14, 10))
    def forward(self, x):
        return self.net(x)
```

**own linear layer：自定义自己的类**

```python
class MyLinear(nn.Module):
    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()
        
        # requires_grad = True
        self.w = nn.Parameter(torch.randn(outp, inp))
        self.b = nn.Parameter(torch.randn(outp))
    def forward(self, x):
        x = x@self.w.t() + self.b
        return x
```

#### 数据增强

**Flip：翻转**

```python
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)
```

**Rotate：旋转**

```python
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   transforms.RandomRotation(15),
                                   transforms.RandomRotation([90, 180, 270]),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)
```

**Scale：缩放**

```python
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   transforms.Resize([32, 32]),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)
```

**Crop：随机裁剪**

```python
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   transforms.RandomCrop([28, 28]),
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)
```

**Noise：噪声，在numpy中提供的**

**GAN**

### 时间序列

#### RNN原理

Naive version：每一个单词都使用一个[wi, bi]去处理

Weight sharing：不同的单词用同一个[w, b]去处理

Consistent memory：保存语境信息

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\RNN-1.png" alt="RNN-1" />
$$
\begin{aligned}
h_t &= f_w(h_{t-1},x_t) \\
\\
h_t &= \tanh{(W_{hh}h_{t-1}+W_{xh}x_t)} \\
y_t &= W_{hy}h_t \qquad \text{这里的$h_t$可以是取任意一次的，得到任意一次的$y_t$} \\
\\
\cfrac{\partial E_t}{\partial W_{hh}} &= \sum_{i=0}^t{\cfrac{\partial E_t}{\partial y_t} \cfrac{\partial y_t}{\partial h_t} \cfrac{\partial h_t}{\partial h_i} \cfrac{\partial h_i}{\partial W_{hh}}} \\
\\
\cfrac{\partial h_t}{\partial h_i} &= \cfrac{\partial h_t}{\partial h_{t-1}} \cfrac{\partial h_{t-1}}{\partial h_{t-2}} \cdots \cfrac{\partial h_{i+1}}{\partial h_i} = \prod_{k=i}^{t-1}{\cfrac{\partial h_{k+1}}{\partial h_k}} \\
\cfrac{\partial h_{k+1}}{\partial h_k} &= diag(f'(W_{xh}x_i+W_{hh}h_{i-1}))W_{hh} \qquad \text{diag是一种对角矩阵的表示方式} \\
\\
\cfrac{\partial h_k}{\partial h_1} &= \prod_i^k{diag(f'(W_{xh}x_i+W_{hh}h_{i-1}))W_{hh}}
\end{aligned}
$$

```python
run = nn.RNN(100:input_size, 10:hidden_size[memory_size], num_layers=1)
run._parameters.keys()

# forward函数
out, ht = forward(x, h0:[layer:层数, b, 10:memory_dim]默认0做为初值开始训练)
```

**RNNCell**

```python
cell1 = nn.RNNCell(100, 30)
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
    h2 = cell2(h1, h2)
print(h2.shape)
```

#### 时间序列预测

**Sample data**

```python
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)
```

**NetWork**

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.run = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        for p in self.run.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.run(x, hidden_prev)
        # [b, seq, h] => [seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)  # [seq, h] => [seq, 1]
        out = out.unsqueeze(dim=0)  # => [1, seq, 1]
        return out, hidden_prev
```

**Train**

```python
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = torch.zeros(1, 1, hidden_size)
for iter in range(6000):
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("Iteration: {} loss {}".format(iter, loss.item()))
```

**Predict**

```python
predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    (pred, hidden_prev) = model(input, hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])
```

#### 梯度弥散和梯度爆炸

**Step1.Gradient Exploding：梯度爆炸**
$$
\begin{aligned}
\hat{g} &\gets \cfrac{\partial \varepsilon}{\partial \theta} \\
\text{if} \quad \Vert \hat{g} \Vert &\ge threshold \quad \text{then} \\
\qquad \hat{g} &\gets \cfrac{threshold}{\Vert \hat{g} \Vert} \hat{g} \\
\end{aligned}
$$

```python
loss = criterion(output, y)
model.zero_grad()
loss.backward()
for p in model.parameters():
    print(p.grad.norm())
    torch.nn.utils.clip_grad_norm_(p, 10)
optimizer.step()
```

### LSTM网络

因为RNN只能记住最近相关的一些语境的单词，之前的可能都会忘记。

LSTM不仅解决梯度弥散问题，还解决了记忆长短的问题。

<img src="F:\文档\Typora Files\markdown-notes\images\notes\python\LSTM-1.png" alt="LSTM-1" />

![LSTM-2](F:\文档\Typora Files\markdown-notes\images\notes\python\LSTM-2.png)![LSTM-3](F:\文档\Typora Files\markdown-notes\images\notes\python\LSTM-3.png)![LSTM-4](F:\文档\Typora Files\markdown-notes\images\notes\python\LSTM-4.png)

| input gate | forget gate | behavior                    |
| ---------- | ----------- | --------------------------- |
| 0          | 1           | remember the previous value |
| 1          | 1           | add to the previous value   |
| 0          | 0           | erase the value             |
| 1          | 0           | overwrite the value         |

$$
\begin{aligned}
\cfrac{\partial C_t}{\partial C_{t-1}} &= \cfrac{\partial C_t}{\partial f_t}\cfrac{\partial f_t}{\partial h_{t-1}}\cfrac{\partial h_{t-1}}{\partial C_{t-1}}+\cfrac{\partial C_t}{\partial i_t}\cfrac{\partial i_t}{\partial h_{t-1}}\cfrac{\partial h_{t-1}}{\partial C_{t-1}} \\ &+\cfrac{\partial C_t}{\partial \widetilde{C}_t}\cfrac{\partial \widetilde{C}_t}{\partial h_{t-1}}\cfrac{\partial h_{t-1}}{\partial C_{t-1}}+\cfrac{\partial C_t}{\partial C_{t-1}} \\
\\
\cfrac{\partial C_t}{\partial C_{t-1}} &= C_{t-1} \sigma'(\cdot)W_f*o_{t-1}\tanh'(C_{t-1}) \\
&+\widetilde{C}_t\sigma'(\cdot)W_i*o_{t-1}\tanh'(C_{t-1}) \\
&+i_t\tanh'(\cdot)W_C*o_{t-1}\tanh'(C_{t-1}) \\
&+f_t
\end{aligned}
$$

```python
#nn.LSTM
lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
# [word num, b, word vec]
x = torch.randn(10, 3, 100)
out, (h, c) = lstm(x)

# nn.LSTMCell
# two layer lstm
cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
h1 = torch.zeros(3, 30)
c1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
c2 = torch.zeros(3, 20)
for xt in x:
    h1, c1 = cell1(xt, [h1, c1])
    h2, c2 = cell2(xt, [h2, c2])
```

### 实战

#### 数据集收集和自定义

##### Step1.Load data

继承`torch.utils.data.Dataset`类

实现\__len__方法

实现\__getitem__方法

```python
# Custom Dataset
class NumbersDataset(Dataset):
    def __init__(self, training=True):
        if training:
            self.samples = train_data
        else:
            self.samples = test_data
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
```

##### Preprocessing：预处理

Image Resize：224x224 for ResNet

Data Argumentation：数据增强

Normalize

ToTensor

##### Step2.build model



##### Step3.Train and Test

```python
for epoch in range(epochs):
    train(train_db)
    if epoch % 10 == 0:
    	val_acc = evalute(model, val_loader)
    	if val_acc > best_acc:
            torch.save(model.state_dict(), 'best.mdl')
        if out_of_patience():
            break
model.load_state_dict(torch.load('best.mdl'))
test_acc = evalute(model, test_loader)
```

##### Step4.Transfer learning

```python
from torchvision.models import resnet18

trained_model = resnet18(pretrained=True)
model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]，得到前17层的数据
                      Flatten(),  # [b, 512, 1, 1] => [b, 512]
                      nn.Linear(512, 5)
                      ).to(device)
```

### AutoEncoder：自动编码器

[降维（visualization）可视化的网站](https://projector.tensorflow.org/)
$$
\begin{aligned}
l_i(\theta,\phi) &= -E_{z \sim q_{\theta}(z|x_i)}[\log p_{\phi}(x_i|z)]+KL(q_{\theta}(z|x_i)||p(z)) \\
KL(P||Q) &= \int_{-\infty}^{+\infty}{p(x)\log \cfrac{p(x)}{q(x)}dx} \qquad \text{这里要求q逼近于p的分布，取得最小值}
\end{aligned}
$$
VAE（变分自编码器），Reparameterization trick

### GAN：生成对抗网络

$$
\begin{aligned}
\max_D L(D,G) &= E_{x \sim p_r(x)}[\log D(x)] + E_{z \sim p_z(z)}[\log (1-D(G(z)))] \\
\min_G L(D,G) &= E_{x \sim p_r(x)}[\log D(x)] + E_{x \sim p_g(x)}[\log (1-D(x))] \\
\\
\\
f(\widetilde{x}) &= A\log \widetilde{x} + B\log(1-\widetilde{x}) \\
\text{set} \quad \cfrac{df(\widetilde{x})}{d\widetilde{x}}&=0 \\
\text{have} \quad D^*(x) &= \widetilde{x}^*=\cfrac{A}{A+B}=\cfrac{p_r(x)}{p_r(x)+p_g(x)}\in [0,1] \\
\\
\\
D_{JS}(p_r||p_g) &= \cfrac{1}{2}D_{KL}(p_r||\cfrac{p_r+p_g}{2})+\cfrac{1}{2}D_{KL}(p_g||\cfrac{p_r+p_g}{2}) \\
&= \cfrac{1}{2}\left( \log2+\int_x{p_r(x)\log \cfrac{p_r(x)}{p_r+p_g(x)}dx} \right) \\
&+ \cfrac{1}{2}\left( \log2+\int_x{p_g(x)\log \cfrac{p_g(x)}{p_r+p_g(x)}dx} \right) \\
&= \cfrac{1}{2}\left( \log4+L(G,D^*) \right)
\end{aligned}
$$

[github上GAN的各种分类](https://github.com/hindupuravinash/the-gan-zoo)













