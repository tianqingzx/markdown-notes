@[toc]

## 优化器



#### Adagrad原理

$s_0=0$为梯度平方累计和

$s=\sum{g_x^2}$

$$
\hat{lr}=\frac{\eta}{\sqrt{s+\epsilon}},\quad \epsilon=10^{-10} \\
x=x-g_x\cdot \hat{lr}
$$
这里的$\eta = lr$

#### RMSprop原理

假设目标函数为$L(x,y)=x^2+10y^2$，平滑常数为$\alpha = 0.9$，$\epsilon = 10^{-6}$，$r_x,r_y=0$

计算出梯度为：$g_x,g_y$
$$
累计梯度的平方：&r_x=\alpha r_x+(1-\alpha)(g_x)^2, \quad r_y=... \\
更新参数：&x=x-g_x\cdot \frac{lr}{\sqrt{r_x}+\epsilon}
$$
其中，若$momentum \neq 0$，则：
$$
\bar{r_x}=\bar{r_x}\cdot momentum+\frac{g_x}{\sqrt{r_x}+\epsilon} \\
x=x-lr\cdot \bar{r_x}
$$

#### Momentum SGD原理

$$
v_t=\gamma v_{t-1}+\eta \nabla_\theta J(\theta_t), \quad \gamma=0.9 \\
\theta_{t+1}=\theta_t-v_t
$$

## 网络trick 小技巧

#### LogSoftmax

**Softmax的缺点**：如果有得分值特别大的情况，即zi特别大，会出现上溢情况；如果zc中很小的负值很多，会出现下溢情况（超出精度范围会向下取0），分母为0，导致计算错误。所以引入了log_softmax。
$$
\begin{align}
\operatorname{LogSoftmax}(x_i) &= \log\left(\frac{e^{x_i-x_m}}{\sum_{j=0}^n e^{x_j-x_m}} \right) \\
&= (x_i-x_m)-\log\left(\sum_{j=0}^n e^{x_j-x_m} \right)
\end{align}
$$
其中$x_m$为输入$X=(x_0,x_1,...,x_n)$中最大的元素；

针对上溢：$x_i-x_m \leq 1$

针对下溢：
$$
\begin{align}
\sum_{j=0}^n e^{x_j-x_m} &= e^{x_0-x_m} + e^{x_1-x_m} + \cdots + e^{x_m-x_m} + \cdots + e^{x_n-x_m} \\
&= e^{x_0-x_m} + e^{x_1-x_m} + \cdots + 1 + \cdots + e^{x_n-x_m} \\
& \gt 1
\end{align}
$$

#### BN与bias

**使用了BN后就不用使用bias，会使得bias的效果被抵消掉。**

BN操作的关键核心可写为：
$$
y_i = \frac{x_i-\bar{x}}{\sqrt{D(x)}}
$$
加上偏置后可写为：
$$
y_i^b = \frac{x_i^b-\bar{x^b}}{\sqrt{D(x^b)}}
$$
其中，
$$
x_i^b = x_i + b
$$
最后对分子、分母部分分别进行化简可得，
$$
y_i^b = y_i
$$
