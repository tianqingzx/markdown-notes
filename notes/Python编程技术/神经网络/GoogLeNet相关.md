#### GoogLeNet详解

网络中的亮点：

+ 引入了Inception结构（融合不同尺度的特征信息）
+ 使用1x1的卷积核进行降维以及映射处理
+ 添加两个辅助分类器帮助训练（因为低层特征也有一定的价值，最后使用一个小的权重就行了）
+ 丢弃全连接层，使用平均池化层（大大减少模型参数）

![GoogLeNet-1](F:\文档\Typora Files\markdown-notes\images\notes\python\GoogLeNet-1.png)

