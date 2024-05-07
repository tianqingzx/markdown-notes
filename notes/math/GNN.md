## GNN
#### GCN 图卷积网络

GCN其实是拉普拉斯平滑的特殊形式。

下式是GCN原论文中的阐述，一个基于分层传播规则的L层GCN，其层与层之间的传播方式为：
$$
H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
$$
$\tilde{A}=A+I_N$，是邻接矩阵，其中$I_N$是一个单位矩阵，为了能够利用上结点自身特征；

$\tilde{D}_{ii}=\sum_j{\tilde{A}_{ij}}$，是$\tilde{A}$的度矩阵；即$\tilde{D}$只在对角线上有值，为对应结点的度；

$W^{(l)}$，是$l$层的权重矩阵；

$H^{(l)}$，是第$l$层的中间输出，$H^{(0)}=X,\quad H^{(L)}=Z$；其中$X \in R^{N \times d}$，$N$为结点个数，$d$为每个结点特征向量的维度；

$L$是指结点特征能够传播的最远距离，$L$一般设置为2到3，堆叠过多的GCN网络可能使得输出特征过度平滑;

这里的拉普拉斯矩阵为$L=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$，这个步骤引入了再正则化（renormalization）技巧；暂时这里只讨论无向无权图。

---

创造一个 Message Passing Networks可以表示成下式：
$$
\mathbf{x}_i^{(k)}=\gamma^{(k)}\left(\mathbf{x}_i^{(k-1)}, \bigoplus_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)}, \mathbf{e}_{j, i}\right)\right)
$$
其中$\bigoplus$代表任意一种聚合算子，例如：add、mean、max。

#### GAT 图注意力网络

输入：$\mathbf{h}=\left\{\vec{h}_1, \vec{h}_2, \ldots, \vec{h}_N\right\}, \vec{h}_i \in \mathbb{R}^F$

输出：$\mathbf{h}^{\prime}=\left\{\vec{h}_1^{\prime}, \vec{h}_2^{\prime}, \ldots, \vec{h}_N^{\prime}\right\}, \vec{h}_i^{\prime} \in \mathbb{R}^{F^{\prime}}$

$a: \mathbb{R}^{F^{\prime}} \times \mathbb{R}^{F^{\prime}} \rightarrow \mathbb{R}$，$a$是attention操作，为一个single-layer feedforward neural network实现的；
$$
e_{i j}=a\left(\mathbf{W} \vec{h}_i, \mathbf{W} \vec{h}_j\right)=a\left( \left[ \mathbf{W} \vec{h}_i \| \mathbf{W} \vec{h}_j \right] \right) \\
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
\vec{h}_i^{\prime}=\mathop{\Big{\|}}\limits_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{i j} \mathbf{W} \vec{h}_j\right)
$$


#### GATv2 动态注意力

GAT(2018)：$e(h_i,h_j)=\operatorname{LeakyReLU}\left(\mathbf{a}^T \cdot \left[\mathbf{W} \vec{h}_i \| \mathbf{W} \vec{h}_j\right]\right)$

GATv2：$e(h_i,h_j)=\mathbf{a}^T \operatorname{LeakyReLU}\left(\mathbf{W} \cdot \left[\vec{h}_i \| \vec{h}_j\right]\right)$

#### HAN 异构图模型

meta-path元路径：连接两个对象的复合关系，例如，结点类型A和结点类型B，A-B-A和B-A-B都是一种元路径。

通过元路径连接在一起的两个同类型结点为相邻结点。

##### 结点级别

$$
\begin{align}
e_{ij}^{\Phi} &= \operatorname{att}_{node}{(\mathbf{h}_i^{\prime},\mathbf{h}_j^{\prime};\Phi)} \\
\alpha_{ij}^{\Phi} &= \operatorname{softmax}_{j}{(e_{ij}^{\Phi})} = \frac{\exp\left(\sigma(\mathbf{a}_{\Phi}^T \cdot [\mathbf{h}_i^{\prime} \| \mathbf{h}_j^{\prime}]) \right)}{\sum_{k \in \mathcal{N}_i^{\Phi}} \exp \left(\sigma(\mathbf{a}_{\Phi}^T \cdot [\mathbf{h}_i^{\prime} \| \mathbf{h}_k^{\prime}]) \right)} \\
\mathbf{z}_i^{\Phi} &= \sigma\left(\sum_{j \in \mathcal{N}_i^{\Phi}} \alpha_{ij}^{\Phi} \cdot \mathbf{h}_j^{\prime} \right)
\end{align}
$$

对于上述第一个公式，这里计算任意结点$i$与其相邻结点$j \in \mathcal{N}_i$之间的注意力系数；

第三个公式，也可以考虑使用多头attention；

##### 语义级别

计算多种meta-path之间的关系，
$$
\begin{align}
(\beta_{\Phi_1}, \dots, \beta_{\Phi_P}) &= \operatorname{att}_{sem}(\mathbf{Z_{\Phi_1}}, \dots, \mathbf{Z_{\Phi_P}}) \\
\nonumber ~\\
\nonumber ~\\
w_{\Phi_i} &= \frac{1}{\vert \mathcal{V} \vert} \sum_{i \in \mathcal{V}}\mathbf{q}^T \cdot \operatorname{tanh}{(\mathbf{W}\cdot \mathbf{z}_i^{\Phi_p} + \mathbf{b})} \\
\beta_{\Phi_p} &= \frac{\exp(w_{\Phi_p})}{\sum_{p=1}^P \exp(w_{\Phi_p})} \\
\mathbf{Z} &= \sum\limits_{p=1}^P \beta_{\Phi_p} \cdot \mathbf{Z}_{\Phi_p}
\end{align}
$$

上述第4个公式中的$\mathbf{Z}_{\Phi_p} = (\mathbf{z}_1^{\Phi_p}, \dots, \mathbf{z}_{\vert \mathcal{V}\vert}^{\Phi_p})$；





