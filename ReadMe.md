# 利用Pytorch实现一些经典的分类网络
在cifar-10数据集上复现一些经典的卷积神经网络。


# VGG
[VGG论文链接](https://arxiv.org/abs/1409.1556)

VGG16相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。
5x5卷积看做一个小的全连接网络在5x5区域滑动，我们可以先用一个3x3的卷积滤波器卷积，然后再用一个全连接层连接这个3x3卷积输出，这个全连接层我们也可以看做一个3x3卷积层。这样我们就可以用两个3x3卷积级联（叠加）起来代替一个 5x5卷积。

VGG网络具体结构图如下：
![VGG网络结构图](https://d2mxuefqeaa7sj.cloudfront.net/s_8C760A111A4204FB24FFC30E04E069BD755C4EEFD62ACBA4B54BBA2A78E13E8C_1491022251600_VGGNet.png)
# ResNet
[ResNet论文链接](https://arxiv.org/pdf/1512.03385.pdf)

随着网络的加深，出现了**训练集**准确率下降的现象，我们可以确定**这不是由于Overfit过拟合造成的**(过拟合的情况训练集应该准确率很高)；所以作者针对这个问题提出了一种全新的网络，叫深度残差网络，它允许网络尽可能的加深，并且不会出现网络的退化现象。

对于一个堆积层结构（几层堆积而成）当输入为时其学习到的特征记为，现在我们希望其可以学习到残差，这样其实原始的学习特征是。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。

![enter link description here](https://img-blog.csdn.net/20180114184946861?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

ResNet提供两种连接的方式：
![enter link description here](https://img-blog.csdn.net/20180114183212429?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

对于resnet18、32使用左边的连接方式，对于深度更深的resnet50、101使用右边的连接方式。
对于不同层级的featuremap会采用某种短路策略上，让输入的featuremap尺寸和输出的一致，从而能够完成 out += input的操作。

不同的resnet网络具体结构图：
![enter image description here](https://img-blog.csdn.net/20180114205444652?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
# MobileNet
[MobileNett论文链接](https://arxiv.org/abs/1704.04861)

MobileNets模型基于深度可分解的卷积，它可以将标准卷积分解成一个深度卷积和一个点卷积（1 × 1卷积核）。深度卷积将每个卷积核应用到每一个通道，而1 × 1卷积用来组合通道卷积的输出。后文证明，这种分解可以有效减少计算量，降低模型大小。下图说明了标准卷积是如何进行分解的。

![分离卷积](https://img-blog.csdn.net/20170425202335040?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

直观上来看，这种分解在效果上确实是等价的。比如，把上图的代号化为实际的数字，输入图片维度是11 × 11 × 3，标准卷积为3 × 3 × 3 ×16（假设stride为2，padding为1），那么可以得到输出为6 × 6 × 16的输出结果。现在输入图片不变，先通过一个维度是3 × 3 × 1 × 3的深度卷积（输入是3通道，这里有3个卷积核，对应着进行计算，理解成for循环），得到6 × 6 × 3的中间输出，然后再通过一个维度是1 × 1 × 3 ×16的1 ×1卷积，同样得到输出为6 × 6 × 16。以上解析还可以借助一幅经典的GIF图来理解，先放这里了。

![enter image description here](https://img-blog.csdn.net/20170816153023677?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVzc2VfTXg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

使用pytorch在实现这个操作的时候，在卷积操作的时候改变 groups 就可以用一个卷积核卷积一组featuremap。从而减少计算量。然后在使用一组1*1的卷积核对，恢复featuremap数量。
具体的mobilenet结构如下：

![enter image description here](https://img-blog.csdn.net/20170425202723997?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
