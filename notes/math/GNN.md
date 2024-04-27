## GNN
#### GCN 图卷积网络

GCN其实是拉普拉斯平滑的特殊形式。

下式是GCN原论文中的阐述，一个基于分层传播规则的L层GCN，其层与层之间的传播方式为：
$$
H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
$$
$\tilde{A}=A+I_N$，是邻接矩阵，其中$I_N$是一个单位矩阵，为了能够利用上结点自身特征；

$\tilde{D}_{ii}=\sum_j{\tilde{A}_{ij}}$，是$\tilde{A}$的度矩阵；

$W^{(l)}$，是$l$层的权重矩阵；

$H^{(l)}$，是第$l$层的中间输出，$H^{(0)}=X,\quad H^{(L)}=Z$；

$L$是指结点特征能够传播的最远距离，$L$一般设置为2到3，堆叠过多的GCN网络可能使得输出特征过度平滑。

---

创造一个 Message Passing Networks可以表示成下式：
$$
\mathbf{x}_i^{(k)}=\gamma^{(k)}\left(\mathbf{x}_i^{(k-1)}, \bigoplus_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)}, \mathbf{e}_{j, i}\right)\right)
$$
其中$\bigoplus$代表任意一种聚合算子，例如：add、mean、max

#### GAT 图注意力网络

输入：$\mathbf{h}=\left\{\vec{h}_1, \vec{h}_2, \ldots, \vec{h}_N\right\}, \vec{h}_i \in \mathbb{R}^F$

输出：$\mathbf{h}^{\prime}=\left\{\vec{h}_1^{\prime}, \vec{h}_2^{\prime}, \ldots, \vec{h}_N^{\prime}\right\}, \vec{h}_i^{\prime} \in \mathbb{R}^{F^{\prime}}$

$a: \mathbb{R}^{F^{\prime}} \times \mathbb{R}^{F^{\prime}} \rightarrow \mathbb{R}$，$a$是attention操作
$$
e_{i j}=a\left(\mathbf{W} \vec{h}_i, \mathbf{W} \vec{h}_j\right) \\
\alpha_{i j}=\operatorname{softmax}_j\left(e_{i j}\right)=\frac{\exp \left(e_{i j}\right)}{\sum_{k \in \mathcal{N}_i} \exp \left(e_{i k}\right)}
$$

+ 其中$\mathcal{N}_i$是结点$i$在图中的邻接点集合

上式可以更具体的描述如下，
$$
\alpha_{i j}=\frac{\exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^T\left[\mathbf{W} \vec{h}_i \| \mathbf{W} \vec{h}_j\right]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^T\left[\mathbf{W} \vec{h}_i \| \mathbf{W} \vec{h}_k\right]\right)\right)}
$$

+ 这里的attention计算使用的是一个单层前馈神经网络
+ LeakyReLU($\alpha=0.2$)
+ $\|$是聚合操作

然后更新参数如下，
$$
\vec{h}_i^{\prime}=\sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{i j} \mathbf{W} \vec{h}_j\right)
$$

+ 这里的$\sigma$表示非线性激活函数

也可以使用多头注意力机制，
$$
\vec{h}_i^{\prime}=\parallel_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{i j} \mathbf{W} \vec{h}_j\right)
$$


#### GATv2 动态注意力

GAT(2018)：$e(h_i,h_j)=\operatorname{LeakyReLU}\left(\mathbf{a}^T \cdot \left[\mathbf{W} \vec{h}_i \| \mathbf{W} \vec{h}_j\right]\right)$

GATv2：$e(h_i,h_j)=\mathbf{a}^T \operatorname{LeakyReLU}\left(\mathbf{W} \cdot \left[\vec{h}_i \| \vec{h}_j\right]\right)$