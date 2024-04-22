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