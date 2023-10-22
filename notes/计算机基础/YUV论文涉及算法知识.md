#### 整体代码流程

加载ResNet50的预训练模型

加载数据以及类别名

定义图形变换，将输入图片resize到299x299大小

使用DataLoader类加载训练数据

定义BIM方法产生对抗性扰动（BIM使用小步多次的梯度更新方案产生多次小扰动，提高攻击能力）

```
atk = torchattacks.BIM(model, eps=4 / 255, alpha=1 / 255, steps=20)
```

Bottleneck 是网络中的最小残差模块

ResNet 网络中有两种 Bottleneck 结构，区别在于是否带一个 conv 卷积层，为了使通道数相同

这里只使用最后一个残差模块保留梯度信息和注意力信息CNN最后一层特征图富含有最为丰富类别语意信息（可以理解为高度抽象的类别特征），因此CAM基于最后一层特征图进行可视化

```
target_layer = model.layer4[-1]
cam = CAM(model=model, target_layer=target_layer, use_cuda=True)
```

这里的CAM策略选择了“gradcam++”结构，克服了CAM不能分析中间层热力图的缺点

使用二值化因子将CAM生成的注意力图进行二值化，因子设置为0.5

使用BIM方法生成对抗样本：

```
adv_images = atk(images, labels)  # 使用BIM方法生成的对抗样本，少量多次更新梯度
```

二值化map==0的地方用原始图像替换对应的对抗样本区域

由替换后的对抗样本矩阵减法减去对应的原始图像数值，由此生成对抗扰动

然后将对抗样本输入网络进行预测

```
outputs = model(adv_images)
```



#### 以下是注释代码的流程

将原始rgb转化为yuv444

For循环最大50次：

使用BIM方法生成对抗样本

```
adv_images = atk(images, labels)
# 将对抗rgb转yuv
adv_yuv = rgb2yuv(adv_images)
```

原始yuv经过map==1的区域替换为对抗yuv，生成的结果为advy_yuv

将y通道的对抗扰动嵌入uv通道：计算原始图像和对抗样本y通道的差值作为嵌入数据

1. 将VU通道拉平，然后原始VU-对抗VU，赋值给err数组
2. 算术编码对嵌入信息进行压缩

```
# 熵编码中的算数编码
code, dic = Arithmetic_encode(err, precision=32)
```

3. 将对抗样本的cbcr通道作为载体图像
4. 使用PEE嵌入，误差扩展算法：embed_data里面前18位存的是压缩后的数据长度（这里会根据嵌入数据计算T值？）

```
# 根据嵌入数据计算T值
T = calculate_threshold(cover, embed_data, m * n)
...
# 进行信息嵌入
Istego = PE_encode(cover, T, embed_data)
...
# 提取原始数据
PE_decode(T_value, img)
...
return '信息嵌入后的图像uv通道', '原始图像的uv通道', '嵌入的数据串', '算术编码返回的dic'
```

使用嵌入信息的uv通道替换上一步生成的对抗yuv的uv通道，进而生成完整的对抗样本

将生成的可逆对抗样本输入模型进行预测：如果模型预测失败则结束循环，否则将advy_yuv作为下一个训练周期的images重新训练对抗样本
