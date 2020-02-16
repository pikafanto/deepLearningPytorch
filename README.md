<!-- TOC -->

- [1 生成数据集并读取数据](#1-生成数据集并读取数据)
- [2 定义模型并初始化模型参数](#2-定义模型并初始化模型参数)
    - [2.1 线性回归模型](#21-线性回归模型)
            - [2.1.0 全连接层](#210-全连接层)
            - [2.1.1 线性回归模型结构](#211-线性回归模型结构)
            - [2.1.2 线性回归模型的定义](#212-线性回归模型的定义)
            - [2.1.3 线性回归模型参数的初始化](#213-线性回归模型参数的初始化)
            - [2.1.4 线性回归模型的前向传播](#214-线性回归模型的前向传播)
    - [2.2 softmax回归模型](#22-softmax回归模型)
            - [2.2.1 softmax回归模型结构](#221-softmax回归模型结构)
            - [2.2.2 softmax回归模型的定义](#222-softmax回归模型的定义)
            - [2.2.3 softmax回归模型参数的初始化](#223-softmax回归模型参数的初始化)
            - [2.2.4 softmax回归模型的前向传播](#224-softmax回归模型的前向传播)
    - [2.3 多层感知机](#23-多层感知机)
            - [2.3.1 多层感知机模型结构](#231-多层感知机模型结构)
            - [2.3.2 多层感知机模型定义](#232-多层感知机模型定义)
            - [2.3.3 多层感知机模型参数的初始化](#233-多层感知机模型参数的初始化)
            - [2.3.4 多层感知机的前向传播](#234-多层感知机的前向传播)
    - [2.4 卷积神经网络(LeNet)](#24-卷积神经网络lenet)
            - [2.4.0 卷积层与池化层](#240-卷积层与池化层)
            - [2.4.1 卷积神经网络模型结构](#241-卷积神经网络模型结构)
            - [2.4.2 卷积神经网络模型的定义](#242-卷积神经网络模型的定义)
    - [2.5 深度卷积神经网络(AlexNet)](#25-深度卷积神经网络alexnet)
            - [2.5.1 深度卷积神经网络模型的结构](#251-深度卷积神经网络模型的结构)
            - [2.5.2 深度卷积神经网络模型的定义](#252-深度卷积神经网络模型的定义)
    - [2.6 VGG网络模型(使用重复元素的网络)](#26-vgg网络模型使用重复元素的网络)
            - [2.6.1 VGG网络模型的结构](#261-vgg网络模型的结构)
            - [2.6.2 VGG网络模型的定义](#262-vgg网络模型的定义)
    - [2.7 NiN模型(网络中的网络)](#27-nin模型网络中的网络)
            - [2.7.1 NiN模型的结构](#271-nin模型的结构)
            - [2.7.2 NiN模型的定义](#272-nin模型的定义)
    - [2.8 GoogLeNet模型(含并行连结的网络)](#28-googlenet模型含并行连结的网络)
            - [2.8.1 GoogLeNet模型结构](#281-googlenet模型结构)
            - [2.8.2 GoogLeNet模型的定义](#282-googlenet模型的定义)
    - [2.9 使用批量归一化层的LeNet](#29-使用批量归一化层的lenet)
            - [2.9.0 批量归一化层](#290-批量归一化层)
            - [2.9.1 使用批量归一化层的LeNet模型结构](#291-使用批量归一化层的lenet模型结构)
            - [2.9.2 使用批量归一化层的LeNet模型的定义](#292-使用批量归一化层的lenet模型的定义)
    - [2.10 残差网络模型(ResNet)](#210-残差网络模型resnet)
            - [2.10.1 残差网络模型结构](#2101-残差网络模型结构)
            - [2.10.2 残差网络模型的定义](#2102-残差网络模型的定义)
    - [2.11 稠密连接网络(DenseNet)](#211-稠密连接网络densenet)
            - [2.11.1 稠密连接网络模型结构](#2111-稠密连接网络模型结构)
            - [2.11.2 稠密连接网络模型的定义](#2112-稠密连接网络模型的定义)
    - [2.12 循环神经网络(RNN)](#212-循环神经网络rnn)
            - [2.12.1 RNN使用示例](#2121-rnn使用示例)
            - [2.12.2 RNN的前向传播过程](#2122-rnn的前向传播过程)
    - [2.13 门控循环单元(GRU)](#213-门控循环单元gru)
            - [2.13.1 GRU使用示例](#2131-gru使用示例)
            - [2.13.2 GRU的前向传播过程](#2132-gru的前向传播过程)
    - [2.14 长短期记忆(LSTM)](#214-长短期记忆lstm)
            - [2.14.1 LSTM的使用示例](#2141-lstm的使用示例)
            - [2.14.2 LSTM的前向传播过程](#2142-lstm的前向传播过程)
    - [2.15 深度循环神经网络](#215-深度循环神经网络)
            - [2.15.1 深度循环神经网络的使用示例](#2151-深度循环神经网络的使用示例)
            - [2.15.2 深度循环神经网络的前向传播](#2152-深度循环神经网络的前向传播)
    - [2.16 双向循环神经网络](#216-双向循环神经网络)
            - [2.16.1 双向循环神经网络的使用示例](#2161-双向循环神经网络的使用示例)
            - [2.16.2 双向循环神经网络的前向传播](#2162-双向循环神经网络的前向传播)
- [3 定义损失函数](#3-定义损失函数)
    - [3.1 均方误差函数](#31-均方误差函数)
    - [3.2 一个包括softmax运算和交叉熵损失计算的函数(CrossEntropyLoss)](#32-一个包括softmax运算和交叉熵损失计算的函数crossentropyloss)
- [4 定义优化算法](#4-定义优化算法)
    - [4.1 小批量随机梯度下降法(SGD)](#41-小批量随机梯度下降法sgd)
    - [4.2 动量法(SGD的另一种使用方式)](#42-动量法sgd的另一种使用方式)
    - [4.3 AdaGrad算法](#43-adagrad算法)
    - [4.4 RMSprop算法](#44-rmsprop算法)
    - [4.5 AdaDelta算法](#45-adadelta算法)
    - [4.6 Adam算法](#46-adam算法)
- [5 训练模型](#5-训练模型)
- [6 应用模型](#6-应用模型)

<!-- /TOC -->
# 1 生成数据集并读取数据
* 具体问题具体分析

# 2 定义模型并初始化模型参数
## 2.1 线性回归模型
#### 2.1.0 全连接层
```Python
# 输入结点个数为256，输出结点个数为128的全连接层
# linear的输入数据维度必须在2维及其以上
linear=torch.nn.Linear(256,128)
# 输入数据x：(64,256)
# 批量大小为64，样本特征个数为256的输入数据，本质上是一个二维张量
x=torch.randn(64,256)
# 输出数据y：(64,128)
# 批量大小为64，样本标签个数为128的输出数据，本质上是一个二维张量
y=linear(x)
```
#### 2.1.1 线性回归模型结构
1. 输入层结点数：num_inputs
2. 输出层结点数：1
3. 输入层与输出层之间全连接
#### 2.1.2 线性回归模型的定义
```Python
net=torch.nn.Sequential(
    # 输入层与输出层之间的全连接网络
    torch.nn.Linear(num_inputs,1)
)
```
#### 2.1.3 线性回归模型参数的初始化
```Python
# 权重参数weight：(1,num_inputs)
torch.nn.init.normal_(net[0].weight,mean=0,std=0.01)
# 偏置值参数bias：(1)
torch.nn.init.constant_(net[0].bias,val=0.0)
```
#### 2.1.4 线性回归模型的前向传播
```Python
# 输入X：(1,num_inputs)
# 输出Y：(1,1)
# Y=torch.mm(X,net[0].weight.T)+net[0].bias
Y=net(X)
```
> 注意：net[0].weight的实际形状为(1,num_inputs)，在进行神经网络模型前向传播时会进行相应的矩阵转置操作。
## 2.2 softmax回归模型
#### 2.2.1 softmax回归模型结构
1. 输入层结点数：num_inputs
2. 输出层结点数：num_outputs
3. 输入层结点与输出层结点全连接
#### 2.2.2 softmax回归模型的定义
```Python
net=torch.nn.Sequential(
    # 输入层与输出层的全连接网络
    torch.nn.Linear(num_inputs,num_outputs)   
)
```
#### 2.2.3 softmax回归模型参数的初始化
```Python
# 权重参数weight：(num_outputs,num_inputs)
torch.nn.init.normal_(net[0].weight,mean=0,std=0.01)
# 偏置值参数bias：(num_outputs)
torch.nn.init.constant_(net[0].bias,val=0.0)
```
#### 2.2.4 softmax回归模型的前向传播
```Python
# 输入X：(1,num_inputs)
# 输出Y：(1,num_outputs)
Y=net(X)
# Y=torch.mm(X,net[0].weight.T)+net[0].bias
```
> 注意：可以将softmax运算包含在损失函数中。
## 2.3 多层感知机
#### 2.3.1 多层感知机模型结构
1. 输入层：结点数为num_inputs
2. 隐藏层：结点数为num_hiddens
3. 输出层：结点数为num_outputs
4. 各层之间全连接
#### 2.3.2 多层感知机模型定义
```Python
# 模型定义
net=torch.nn.Sequential(
    # 输入层与隐藏层之间的全连接网络
    torch.nn.Linear(num_inputs,num_hiddens),
    # 对隐藏层的输出进行激活
    # 采取激活函数ReLU
    torch.nn.ReLU(),
    # 隐藏层与输出层之间的全连接网络
    torch.nn.Linear(num_hiddens,num_outputs)
)
```
#### 2.3.3 多层感知机模型参数的初始化
```Python
for params in net.parameters():
    torch.nn.init.normal_(params, mean=0, std=0.01)
```
#### 2.3.4 多层感知机的前向传播
```Python
# 输入X：(1,num_inputs)
# 输出Y：(1,num_outputs)
Y=net(X)
# t1=torch.mm(X,net[0].weight.T)+net[0].bias
# t2=torch.net[1](t1)，即t2=torch.nn.ReLU(t1)
# Y=torch.mm(t2,net[2].weight.T)+net[1].bias
```
## 2.4 卷积神经网络(LeNet)
#### 2.4.0 卷积层与池化层
```python
# 输入通道为2，输出通道为4，形状为(3,3)，步幅为1，填充为(0,0)的卷积层
# conv2d的输入数据必须在4维及其以上
conv2d=torch.nn.Conv2d(2,4,3,1,0)
# 输入数据x：(256,2,9,9)
# 批量大小为256，输入通道为2，形状为(9,9)的输入数据x，本质上是一个四维张量
x=torch.randn(256,2,9,9)
# 输出数据y：(256,4,7,7)
# 批量大小为256，输出通道为4，形状为(7,7)的输出数据y，本质上是一个四维张量
y=conv2d(x)
```
```Python
# 形状为(2,2)，步幅为2的最大池化层
# maxPool2d的输入数据必须在2维及其以上
maxPool2d=torch.nn.MaxPool2d(2,2)
# 输入数据x：(256,4,8,8)
# 批量大小为256，输出通道为4，形状为(8,8)的输入数据x，本质上是一个四维张量
x=torch.randn(256,4,8,8)
# 输出数据y：(256,4,4,4)
# 批量大小为256，输出通道为4，形状为(4,4)的输出数据y，本质上是一个四维张量
y=maxPool2d(x)
```
#### 2.4.1 卷积神经网络模型结构
* 卷积层块：
    1. (1,6,(5,5))的卷积层：输入通道为1，输出通道为6，形状为(5,5)的卷积层。
    2. sigmoid激活函数。
    3. ((2,2),2)的最大池化层：形状为(2,2)，步幅为2的最大池化层。
    4. (6,16,(5,5))的卷积层：输入通道为6，输出通道为16，形状为(5,5)的卷积层。
    5. sigmoid激活函数。
    6. ((2,2),2)的最大池化层：形状为(2,2)，步幅为2的最大池化层。
* 全连接层块：
    1. (16\*4\*4,120)的全连接层：输入结点个数为16\*4\*4，输出结点个数为120的全连接层。
    2. sigmoid激活函数。
    3. (120,84)的全连接层：输入结点个数为120，输出结点个数为84全连接层。
    4. sigmodi激活函数。
    5. (84,10)的全连接层：输入结点个数为84，输出结点个数为10的全连接层。
#### 2.4.2 卷积神经网络模型的定义
```Python
# LeNet模型
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # 卷积层块conv
        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(1,6,5),
            torch.nn.Sigmodi(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(6,16,5),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2,2)
        )
        # 全连接层块fc
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(16*4*4,120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(120,84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(84,10)
        )
    
    # 前向传播
    def forward(self,img):
        feature=self.conv(img)
        output=self.fc(feature.view(feature.shape[0],-1))
        return output
```
## 2.5 深度卷积神经网络(AlexNet)
#### 2.5.1 深度卷积神经网络模型的结构
* 卷积层块：
    1. (1,96,(11,11),4,(0,0))的卷积层：输入通道为1，输出通道为96，形状为(11,11)，步幅为4，填充为(0,0)的卷积层。
    2. ReLU激活函数。
    3. ((3,3),2)的最大池化层：形状为(3,3)，步幅为2的池化层。
    4. (96,256,(5,5),1,(2,2))的卷积层：输入通道为96，输出通道为256，形状为(5,5)，步幅为1，填充为(2,2)的卷积层。
    5. ReLU激活函数。
    6. ((3,3),2)的最大池化层：形状为(3,3)，步幅为2的池化层。
    7. (256,384,(3,3),1,(1,1))的卷积层：输入通道为256，输出通道为384，形状为(3,3)，步幅为1，填充为(1,1)的卷积层。 
    8. ReLU激活函数。
    9. (384,384,(3,3),1,(1,1))的卷积层：输入通道为384，输出通道为384，形状为(3,3)，步幅为1，填充为(1,1)的卷积层。 
    10. ReLU激活函数。
    11. (384,256,(3,3),1,(1,1))的卷积层：输入通道为384，输出通道为256，形状为(3,3)，步幅为1，填充为(1,1)的卷积层。 
    12. ReLU激活函数。 
    13. ((3,3),2)的最大池化层：形状为(3,3)，步幅为2的池化层。
* 全连接层块：
    1. (256\*5\*5,4096)的全连接层。
    2. ReLU激活函数。
    3. (0.5)的丢弃层：丢弃率为0.5的丢弃层。
    4. (4096,4096)的全连接层。
    5. ReLU激活函数。
    6. (0.5)的丢弃层：丢弃率为0.5的丢弃层。
    7. (4096,10)的全连接层。
#### 2.5.2 深度卷积神经网络模型的定义
```Python
# AlexNet模型
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        # 卷积层块conv
        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(1,96,11,4,0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(96,256,5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(256,384,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,384,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
        )
        # 全连接层块fc
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(256*5*5,4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096,10),
        )
    
    # 前向传播
    def forward(self,img):
        feature=self.conv(img)
        output=self.fc(feature.view(feature.shape[0],-1))
        return output
```
## 2.6 VGG网络模型(使用重复元素的网络)
#### 2.6.1 VGG网络模型的结构
* vgg块（含5个卷积层块）
    1. 卷积层块1（含1个卷积层）
        * (1,64,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * ((2,2),2)的最大池化层。
    2. 卷积层块2（含1个卷积层）
        * (64,128,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * ((2,2),2)的最大池化层。
    3. 卷积层块3（含2个卷积层）
        * (128,256,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * (256,256,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * ((2,2),2)的最大池化层。
    4. 卷积层块4（含2个卷积层）
        * (256,512,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * (512,512,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * ((2,2),2)的最大池化层。
    5. 卷积层块5（含2个卷积层）
        * (512,512,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * (512,512,(3,3),1,(1,1))的卷积层。
        * ReLU激活函数。
        * ((2,2),2)的最大池化层。
* 全连接层块：
    1. (512\*7\*7,4096)的全连接层。
    2. ReLU激活函数。
    3. (0.5)的丢弃层。
    4. (4096,4096)的全连接层。
    5. ReLU激活函数。
    6. (0.5)的丢弃层。
    7. (4096,10)的全连接层。
#### 2.6.2 VGG网络模型的定义
```Python
# VGG网络模型
class VGG(torch.nn.module):
    def __init__(self):
        super(VGG,self).__init__()
        # vgg块
        self.vgg=torch.nn.Sequential()
        self.vgg.add_module(convBlock1,self.getConvBlock(1,1,64))
        self.vgg.add_module(convBlock2,self.getConvBlock(1,64,128))
        self.vgg.add_module(convBlock3,self.getConvBlock(2,128,256))
        self.vgg.add_module(convBlock1,self.getConvBlock(2,256,512))
        self.vgg.add_module(convBlock1,self.getConvBlock(2,512,512))
        # 全连接层块fc
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(512*7*7,4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096,10)
        )
    
    # 建立卷积层块的操作
    def getConvBlock(convNums,inChannels,outChannels):
        convBlock=[]
        for i in convNums:
            convBlock.append(torch.nn.Conv2d(in_Channels,outChannels,3,1,1))
            in_Channels=outChannels
            convBlock.append(torch.nn.ReLU())
        convBlock.append(torch.nn.MaxPool2d(2,2))
        return torch.nn.Sequential(*convBlock
    
    # 前向传播
    def forward(self,img):
        feature=self.vgg(img)
        output=self.fc(feature.view(feature.shape[0],-1))
        return output
```
## 2.7 NiN模型(网络中的网络)
#### 2.7.1 NiN模型的结构
* (1,96,(11,11),4,(0,0))的NiN块：输入通道为1，输出通道为96，形状为(11,11)，步幅为4，填充为(0,0)的NiN块。
    1. (1,96,(11,11),4,(0,0))的卷积层。
    2. ReLU激活函数。
    3. (96,96,(1,1),0,(0,0))的卷积层。
    4. ReLU激活函数。
    5. (96,96,(1,1),0,(0,0))的卷积层。
    6. ReLU激活函数。
* ((3,3),2)的最大池化层。
* (96,256,(5,5),1,(2,2))的NiN块：输入通道为96，输出通道为256，形状为(5,5)，步幅为1，填充为(2,2)的NiN块。
    1. (96,256,(5,5),1,(2,2))的卷积层。
    2. ReLU激活函数。
    3. (256,256,(1,1),0,(0,0))的卷积层。
    4. ReLU激活函数。
    5. (256,256,(1,1),0,(0,0))的卷积层。
    6. ReLU激活函数。
* ((3,3),2)的最大池化层。
* (256,384,(3,3),1,(1,1))的NiN块：输入通道为256，输出通道为384，形状为(3,3)，步幅为1，填充为(1,1)的NiN块。
    1. (256,384,(3,3),1,(1,1))的卷积层。
    2. ReLU激活函数。
    3. (384,384,(1,1),0,(0,0))的卷积层。
    4. ReLU激活函数。
    5. (384,384,(1,1),0,(0,0))的卷积层。
    6. ReLU激活函数。
* ((3,3),2)的最大池化层。
* (0.5)的丢弃层。
* (384,10,(3,3),1,(1,1))的NiN块：输入通道为384，输出通道为10，形状为(3,3)，步幅为1，填充为(1,1)的NiN块。
    1. (384,10,(3,3),1,(1,1))的卷积层。
    2. ReLU激活函数。
    3. (10,10,(1,1),0,(0,0))的卷积层。
    4. ReLU激活函数。
    5. (10,10,(1,1),0,(0,0))的卷积层。
    6. ReLU激活函数。
* 形状与输入数据最后两个维度相同的平均池化层。
#### 2.7.2 NiN模型的定义
```Python
# NiN模型
class NiN(torch.nn.Module):
    def __init__(self):
        super(NiN,self).__init__()
        self.net=torch.nn.Sequential(
            self.getNiNBlock(1,96,11,4,0),
            torch.nn.MaxPool2d(3,2),
            self.getNiNBlock(96,256,5,1,2),
            torch.nn.MaxPool2d(3,2),
            self.getNiNBlock(256,384,3,1,1),
            torch.nn.MaxPool2d(3,2),
            torch.nn.Dropout(0.5),
            self.getNiNBlock(384,10,3,1,1),
            GlobalAvgPool2d(),
        )
    
    def getNiNBlock(self,inCh,outCh,shape,stride,pad):
        return torch.nn.Sequential(
            torch.nn.Conv2d(inCh,outCh,shape,stride,pad)
            torch.nn.ReLU()
            torch.nn.Conv2d(outCh,outCh,1,0,0)
            torch.nn.ReLU()
            torch.nn.Conv2d(outCh,outCh,1,0,0)
            torch.nn.ReLU()
        )
    
    def forward(self,img):
        output=self.net(img)
        return output.view(-1,10)
```
## 2.8 GoogLeNet模型(含并行连结的网络)
#### 2.8.1 GoogLeNet模型结构
* 模块1：
    1. (1,64,(7,7),2,(3,3))的卷积层。
    2. ReLU激活函数
    3. ((3,3),2,(1,1))的最大池化层：形状为(3,3)，步长为2，填充为(1,1)的最大池化层。
* 模块2：
    1. (64,64,(1,1),1,(0,0))的卷积层。
    2. (64,192,(3,3),1,(1,1))的卷积层。
    3. ((3,3),2,(1,1))的最大池化层：形状为(3,3)，步长为2，填充为(1,1)的最大池化层。
* 模块3：
    1. (192,64,(96,128),(16,32),32)的Inception块：以下4条线路的并行连结。
        * 线路1：
            1. (192,64,(1,1),1,(0,0))的卷积层。
            2. ReLU激活函数。
        * 线路2：
            1. (192,96,(1,1),1,(0,0))的卷积层。
            2. (96,128,(3,3),1,(1,1))的卷积层。
            3. ReLU激活函数。
        * 线路3：
            1. (192,16,(1,1),1,(0,0))的卷积层。
            2. (16,32,(5,5),1,(2,2))的卷积层。    
            3. ReLU激活函数。
        * 线路4：
            1. ((3,3),1,(1,1))的最大池化层。
            2. (192,32,(1,1),1,(0,0))的卷积层。
            3. ReLU激活函数。
    2. (256,128,(128,192),(32,96),64)的Inception块。
    3. ((3,3),2,(1,1))的最大池化层。
* 模块4：
    1. (480,192,(96,208),(16,48),64)的Inception块。
    2. (512,160,(112,224),(24,64),64)的Inception块。
    3. (512,128,(128,256),(24,64),64)的Inception块。
    4. (512,112,(144,288),(32,64),64)的Inception块。
    5. (528,256,(160,320),(32,128),128)的Inception块。
    6. ((3,3),2,(1,1))的最大池化层。
* 模块5：
    1. (832,256,(160,320),(32,128),128)的Inception块。
    2. (832,384,(192,384),(48,128),128)的Inception块。
    3. 形状与输出数据最后两维形状相同的平均池化层。
    4. 输出数据形状转换。
    5. (1024,10)的全连接层。
#### 2.8.2 GoogLeNet模型的定义
```Python
class Inception(torch.nn.Module):
    def __init__(self,ci,c1,c2,c3,c4):
        super(Inception,self).__init__()
        # 线路1
        self.l1=torch.nn.Sequential(
            torch.nn.Conv2d(ci,c1,1,1,0),
            torch.nn.ReLU()
        )
        # 线路2
        self.l2=torch.nn.Sequential(
            torch.nn.Conv2d(ci,c2[0],1,1,0),
            torch.nn.Conv2d(c2[0],c2[1],3,1,1),
            torch.nn.ReLU()
        )
        # 线路3
        self.l3=torch.nn.Sequential(
            torch.nn.Conv2d(ci,c3[0],1,1,0),
            torch.nn.Conv2d(c3[0],c3[1],5,1,2),
            torch.nn.ReLU()
        )
        # 线路4
        self.l4=torch.nn.Sequential(
            torch.nn.MaxPool2d(3,1,1),
            torch.nn.Conv2d(ci,c4,1,1,0),
            torch.nn.ReLU()
        )
    
    # 前向传播
    def forward(self,x):
        r1=self.l1(x)
        r2=self.l2(x)
        r3=self.l3(x)
        r4=self.l4(x)
        # 在通道维上对四条线路上的输出进行连结
        return torch.cat((r1,r2,r3,r4),dim=1)
```
```Python
class GoogLeNet(torch.nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        # 模块1
        self.b1=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,7,2,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2,1)
        )
        # 模块2
        self.b2=torch.nn.Sequential(
            torch.nn.Conv2d(64,64,1,1,0),
            torch.nn.Conv2d(64,192,3,1,1),
            torch.nn.MaxPool2d(3,2,1)
        )
        # 模块3
        self.b3=torch.nn.Sequential(
            Inception(192,64,(96,128),(16,32),32)
            Inception(256,128,(128,192),(32,96),64)
            torch.nn.MaxPool2d(3,2,1)
        )
        # 模块4
        self.b4=torch.nn.Sequential(
            Inception(480,192,(96,208),(16,48),64)
            Inception(512,160,(112,224),(24,64),64)
            Inception(512,128,(128,256),(24,64),64)
            Inception(512,512,(144,288),(32,64),64)
            Inception(528,256,(160,320),(32,128),128)
            torch.nn.MaxPool2d(3,2,1)
        )
        # 模块5
        self.b5=torch.nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            GlobalAvgPool2d(),
            FlattenLayer(),
            troch.nn.Linear(1024,10)
        )
    
    # 前向传播
    def forward(self,x):
        r1=self.b1(x)
        r2=self.b2(x)
        r3=self.b3(x)
        r4=self.b4(x)
        r5=self.b5(x)
        return r5
```
## 2.9 使用批量归一化层的LeNet
#### 2.9.0 批量归一化层
```Python
# (16)的一维批量归一化层：应用于输出结点为16的全连接层。
# bn1d的输入数据必须是2维。
bn1d=torch.nn.BatchNorm1d(16)
# 输入数据x：(256,16)
# 批量大小为256，样本标签个数为16的输入数据，本质上是一个二维张量。
x=torch.randn(256,16)
# 输出数据y：(256,16)
# 批量大小为256，样本标签个数为16的输出数据，本质上是一个二维张量。
y=bn1d(x)
# bn1d前向传播过程等价于以下过程
# x在批量大小维度上的均值u
# u=x.mean(dim=0)
# x在批量大小维度上的方差d
# d=((x-u)**2)/256
# 输出数据y
# y=(x-u)/(d**(1/2)+0.00000001)*r+b
# 其中r,b为一维批量归一化层的可学习参数，r初始化为1，b初始化为0
```
```Python
# (16)的二维批量归一化层：应用于输出通道为16的卷积层。
# bn2d的输入数据必须是4维
bn2d=torch.nn.BatchNorm2d(16)
# 输入数据x：(256,16,8,8)
# 批量大小为256，输出通道为16，形状为(8,8)的输入数据，本质上是一个四维张量
x=torch.randn(256,16,8,8)
# 输出数据y：(256,16,8,8)
# 批量大小为256，输出通道为16，形状为(8,8)的输出数据，本质上是一个四维张量
y=bn2d(x)
# bn2d前向传播过程等价于以下过程
# x在输出通道维度上的均值u
# u=x.mean(dim=1)
# x在输出通道维度上的方差d
# d=((x-u)**2)/16
# 输出数据y
# y=(x-u)/(d**(1/2)+0.00000001)*r+b
# 其中r,b为二维批量归一化层的可学习参数，r初始化为1，b初始化为0
```
#### 2.9.1 使用批量归一化层的LeNet模型结构
* 卷积层块：
    1. (1,6,(5,5),1,(0,0))的卷积层。
    2. (6)的二维批量归一化层：应用于输出通道为6的卷积层。
    3. sigmoid激活函数。
    4. ((2,2),2)的最大池化层。
    5. (6,16,(5,5),1,(0,0))的卷积层。
    6. (16)的二维批量归一化层：应用于输出通道为16的卷积层。
    7. sigmoid激活函数。
    8. ((2,2),2)的最大池化层。
* 全连接层块：
    1. (16\*4\*4,120)的全连接层。
    2. (120)的一维批量归一化层：应用于输出结点个数为120的全连接层。
    3. sigmoid激活函数。
    4. (120,84)的全连接层。
    5. (84)的一维批量归一化层：应用于输出结点个数为84的批量归一化层。
    6. sigmoid激活函数。
    7. (84,10)的全连接层。
#### 2.9.2 使用批量归一化层的LeNet模型的定义
```Python
# BNLeNet模型
class BNLeNet(torch.nn.Module):
    def __init__(self):
        super(BNLeNet,self).__init__()
        # 卷积层块
        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(1,6,5,1,0),
            torch.nn.BatchNorm2d(6),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(6,16,5,1,0),
            torch.nn.BatchNorm2d(16),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2,2)
        )
        # 全连接层块
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(16*4*4,120),
            torch.nn.BatchNorm1d(120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(120,84),
            torch.nn.BatchNorm1d(84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(84,10)
        )

    # 前向传播
    def forward(self,x):
        convOutput=self.conv(x)
        fcInput=convOutput.view(-1,256)
        output=self.fc(fcInput)
        return output          
```
## 2.10 残差网络模型(ResNet)
#### 2.10.1 残差网络模型结构
* 模块1：卷积层块。
    1. (1,64,(7,7),2,(3,3))的卷积层。
    2. (64)的二维批量归一化层。
    3. ReLU激活函数。
    4. ((3,3),2,(1,1))的最大池化层。
* 模块2：(64,64,True)残差块：输入通道为64，输出通道为64的特殊残差块。
    1. (64,64,True)的残差块第一小块：
        * 线路1：
            1. (64,64,(3,3),1,(1,1))的卷积层。
            2. (64)的二维批量归一化层。
            3. ReLU激活函数。
            4. (64,64,(3,3),1,(1,1))的卷积层。
            5. (64)的二维批量归一化层。
        * 线路2：
            1. None（特殊之处）
        * 线路1的输出数据 + 线路2的输出数据
    2. (64,64,True)的残差块第二小块：
        * 线路1：
            1. (64,64,(3,3),1,(1,1))的卷积层。
            2. (64)的二维批量归一化层。
            3. ReLU激活函数。
            4. (64,64,(3,3),1,(1,1))的卷积层。
            5. (64)的二维批量归一化层。
        * 线路2：
            1. None
        * 线路1的输出数据 + 线路2的输出数据。
* 模块3：(64,128,False)的残差块：输入通道为64，输出通道为128的残差块。
    1. (64,128,False)的残差块第一小块：
        * 线路1：
            1. (64,128,(3,3),1,(1,1))的卷积层。
            2. (128)的二维批量归一化层。
            3. ReLU激活函数。
            4. (128,128,(3,3),1,(1,1))的卷积层。
            5. (128)的二维批量归一化层。
        * 线路2：
            1. (64,128,(1,1),1,(0,0))的卷积层。
        * 线路1的输出数据 + 线路2的输出数据。
    2. (64,128,False)的残差块第二小块：
        * 线路1：
            1. (128,128,(3,3),1,(1,1))的卷积层。
            2. (128)的二维批量归一化层。
            3. ReLU激活函数。
            4. (128,128,(3,3),1,(1,1))的卷积层。
            5. (128)的二维批量归一化层。
        * 线路2：
            1. None
        * 线路1的输出数据 + 线路2的输出数据。
* 模块4：(128,256,False)的残差块。
* 模块5：(256,512,False)的残差块。
* 模块6：全连接层块。
    1. 输入数据形状与模块5输出数据后两维形状相同的平均池化层。
    2. 张量形状转换层。
    3. (512,10)的全连接层。
#### 2.10.2 残差网络模型的定义
```Python
# ResNetBlock模型
class ResNetBlock(torch.nn.Module):
    def __init__(self,inputChannels,outputChannels,firstBlock=False):
        super(ResNetBlock,self).__init__()
        # 第1小块线路1
        self.block1line1=torch.nn.Sequential(
            torch.nn.Conv2d(inputChannels,outputChannels,3,2,1),
            torch.nn.BatchNorm2d(outputChannels),
            torch.nn.ReLU()
            torch.nn.Conv2d(outputChannels,outputChannels,3,1,1),
            torch.nn.BatchNorm2d(outputChannels)
        )
        # 第1小块线路2
        self.block1line2=None
        if(firstBlock):
            self.block1line2=torch.nn.Conv2d(inputChannels,outputChannels,1,1,0)
        # 第2小块线路1：与第1小块线路1相同
        self.block2line1=self.block1line1
        # 第2小块线路2
        self.block2line2=None
    
    # 前向传播
    def forward(self,x):
        r11=self.block1line1(x)
        r12=self.block1line2(x)
        r1=r11+r12
        r21=self.block2line1(r1)
        r22=self.block2line2(r1)
        r2=r21+r22
        return r2
```
```Python
# ResNet模型
class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        # 模块1：卷积层块
        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,7,2,3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2,1)
        )
        # 模块2：(64,64,True)的残差块
        self.ResNetBlock1=ResNetBlock(64,64,True)
        # 模块3：(64,128,False)的残差块
        self.ResNetBlock2=ResNetBlock(64,128,False)
        # 模块4：(128,256,False)的残差块
        self.ResNetBlock3=ResNetBlock(128,256,False)
        # 模块5：(256,512,False)的残差块
        self.ResNetBlock4=ResNetBlock(256,512,False)
        # 模块6：全连接层块
        self.fc=torch.nn.Sequential(
            GlobalAvgPool2d(),
            FlattenLayer(),
            torch.nn.Linear(512,10)
        )
    
    # 前先向传播
    def forward(self,x):
        r1=self.conv(x)
        r2=self.ResNetBlock1(r1)
        r3=self.ResNetBlock2(r2)
        r4=self.ResNetBlock3(r3)
        r5=self.ResNetBlock4(r4)
        r6=self.fc(r5)
        return r6
```
## 2.11 稠密连接网络(DenseNet)
#### 2.11.1 稠密连接网络模型结构
* 模块1：卷积层块
    1. (1,64,(7,7),2,(3,3))的卷积层。
    2. (64)的批量归一化层。
    3. ReLU激活函数。
    4. ((3,3),2,(1,1))的最大池化层。
* 模块2：稠密块+过渡层
    1. (4,64,32)的稠密块：卷积层块数为4，输入通道为64，输出通道为32的稠密块。
        * (64,32)的卷积层块：线路1与线路2在通道维上的连结。
            1. 线路1：
                * (64)的二维批量归一化层。
                * ReLU激活函数。
                * (64,32,(3,3),1,(1,1))的卷积层。
            2. 线路2：None。
        * (64+32,32)的卷积层块：线路1与线路2在通道维上的连结。
            1. 线路1：
                * (96)的二维批量归一化层。
                * ReLU激活函数。
                * (96,32,(3,3),1,(1,1))的卷积层。
            2. 线路2：None。
        * (96+32,32)的卷积层块：线路1与线路2在通道维上的连结。
            1. 线路1：
                * (128)的二维批量归一化层。
                * ReLU激活函数。
                * (128,32,(3,3),1,(1,1))的卷积层。
            2. 线路2：None。
        * (128+32,32)的卷积层块：线路1与线路2在通道维上的连结。
            1. 线路1：
                * (160)的二维批量归一化层。
                * ReLU激活函数。
                * (160,32,((3,3),1,(1,1))的卷积层。
            2.  线路2：None。
    2. (192,96)的过渡层：输入通道为192，输出通道为96的过渡层。（使得通道数减半）
        * (192)的二维批量归一化层。
        * ReLU激活函数。
        * (192,96,(1,1),1,(0,0))的卷积层。
        * ((2,2),2)的平均池化层：形状为(2,2)，步幅为2的平均池化层。
* 模块3：稠密块+过渡层
    1. (4,96,32)的稠密块：卷积层块数为4，输入通道为96，输出通道为32的稠密块。
    2. (224,112)的过渡层：输入通道为224，输出通道为112的过渡层。（使得通道数减半）
* 模块4：稠密块+过渡层
    1. (4,112,32)的稠密块：卷积层块数为4，输入通道为112，输出通道为32的稠密块。
    2. (240,120)的过渡层：输入通道为240，输出通道为120的过渡层。
* 模块5：稠密块（最后一个稠密块不加过渡层）
    1. (4,120,32)的稠密块：卷积层块数为4，输入通道为112，输出通道为32的稠密块
* 模块6：全连接层块
    1. (248)的二维批量归一化层。
    2. ReLU激活函数。
    3. 平均池化层。
    4. 张量形状变换层。
    5. (248,10)的全连接层。
#### 2.11.2 稠密连接网络模型的定义
```Python
# TrasitionLayer模型
class TrasitionLayer(torch.nn.Module):
    def __init__(self,inChannels,outChannels):
        super(TrasitionLayer,self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.BatchNorm2d(inChannels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(inChannels,outChannels,1,1,0)
            GlobalAvgPool2d(2,2)
        )
    # 前向传播
    def forward(self,x):
        return self.net(x)
```
```Python
# ConvBlock模型
class ConvBlock(torch.nn.Module):
    def __init__(self,inChannels,outChannels):
        super(ConvBlock,self).__init__()
        # 线路1
        self.line1=torch.nn.Sequential(
            torch.nn.BatchNorm2d(inChannels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,32,3,1,1)
        )
        # 线路2：None
    # 前向传播
    def forward(self,x):
        y=self.line1(x)
        # 在通道维上将输入和输出连结
        return torch.cat((x,y),dim=1)
```
```Python
# DenseNetBlock模型
class DenseNetBlock(torch.nn.Module):
    def __init__(self,convNums,inChannels,outChannels):
        convBlocksList=[]
        # convNums个卷积层块
        for i in convNums:
            self.convBlocks.append(ConvBlock(inChannels+i*outChannels,outChannels))
        # 使用ModuleList是为了将convBlock的参数作为当前模型的参数
        self.convBlocks=torch.nn.ModuleList(convBlocksList)
    # 前向传播
    def forward(self,x):
        for convBlock in convBlocks:
            x=convBlock(x)
        return x
```
```Python
# DenseNet模型
class DenseNet(torch.nn.Module):
    def __init__(self):
        super(DenseNet,self).__init__()
        # 模块1：卷积层块
        self.model1=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,7,2,3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2,1)
        )
        # 模块2：稠密块+过渡层
        self.model2=torch.nn.Sequential(
            DenseNetBlock(4,64,32),
            TrasitionLayer(192,96)
        )
        # 模块3：稠密块+过渡层
        self.model3=torch.nn.Sequential(
            DenseNetBlock(4,96,32),
            TrasitionLayer(224,112)
        )
        # 模块4：稠密块+过渡层
        self.model2=torch.nn.Sequential(
            DenseNetBlock(4,112,32),
            TrasitionLayer(240,120)
        )
        # 模块5：稠密块+过渡层
        self.model5=DenseNetBlock(4,120,32)
        # 模块6：全连接层块
        self.model6=torch.nn.Sequential(
            torch.nn.BatchNorm2d(248),
            torch.nn.ReLU(),
            GlobalAvgPool2d(),
            FlattenLayer(),
            torch.nn.Linear(248,10)
        )

    # 前向传播
    def forward(self,x):
        r1=self.model1(x)
        r2=self.model2(r1)
        r3=self.model3(r2)
        r4=self.model4(r3)
        r5=self.model5(r4)
        r6=self.model6(r5)
        return r6
```
## 2.12 循环神经网络(RNN)
#### 2.12.1 RNN使用示例
```Python
# (3,4,1)的循环神经网络：输入特征数为3，隐藏单元个数为4，隐藏层数为1的循环神经网络
rnn=torch.nn.RNN(3,4,1)
# (2,8,3)的输入数据x：时间步数为2，批量大小为8，特征数为3的输入数据，本质上是一个三维张量
# (1,8,4)的初始隐藏状态h0：隐藏层数为1，批量大小为8，隐藏单元个数为4的初始隐藏状态，本质上是一个三维张量
x=torch.randn(2,8,3)
h0=torch.randn(1,8,4)
# (2,8,4)的输出数据y：时间步数为2，批量大小为8，隐藏单元个数为4的输出数据，本质上是一个三维张量
# (1,8,4)的最终隐藏状态h2：隐藏层数为1，批量大小为8，隐藏单元个数为4的最终隐藏状态（即第2时间步的隐藏状态），本质上是一个三维张量
y,h2=rnn(x,h0)
```
#### 2.12.2 RNN的前向传播过程
```Python
# 前向传播过程等价于以下计算过程
tanh=torch.nn.Tanh()
ih1=torch.matmul(x[0],rnn.weight_ih_l0.T)+rnn.bias_ih_l0
hh1=torch.matmul(h0,rnn.weight_hh_l0.T)+rnn.bias_hh_l0
h1=tanh(ih1+hh1)
ih2=torch.matmul(x[1],rnn.weight_ih_l0.T)+rnn.bias_ih_l0
hh2=torch.matmul(t1,rnn.weight_hh_l0.T)+rnn.bias_hh_l0
h2=tanh(ih2+hh2)
y=torch.cat((h1,h2),dim=0)
y,h2
```
## 2.13 门控循环单元(GRU)
#### 2.13.1 GRU使用示例
```Python
# (4,3,1)的门控循环单元：输入特征个数为4，隐藏单元个数为3，隐藏层数为1的门控循环单元
gru=nn.GRU(4,3,1)
# (2,8,4)的输入数据x：时间步长为2，批量大小为8，输入特征个数为4的输入数据
# (1,8,3)的初始隐藏状态h0：隐藏层数为1，批量大小为8，隐藏单元个数为3的初始隐藏状态
x=torch.randn(2,8,4)
h0=torch.randn(1,8,3)
# (2,8,3)的输出数据y：时间步长为2，批量大小为8，隐藏单元个数为3的输出数据
# (1,8,3)的最终隐藏状态h2：隐藏层数为1，批量大小为8，隐藏单元个数为3的最终隐藏状态（即第2时间步的隐藏状态）
y,h2=gru(x,h0)
y,h2
```
#### 2.13.2 GRU的前向传播过程
```Python
# 前向传播的计算过程
tanh=nn.Tanh()
sigmoid=nn.Sigmoid()
# 第1时间步的重置门
r1_ir=torch.matmul(x[0],gru.weight_ih_l0[0:3].T)+gru.bias_ih_l0[0:3]
r1_hr=torch.matmul(h0,gru.weight_hh_l0[0:3].T)+gru.bias_hh_l0[0:3]
r1=sigmoid(r1_ir+r1_hr)
# 第1时间步的更新门
z1_iz=torch.matmul(x[0],gru.weight_ih_l0[3:6].T)+gru.bias_ih_l0[3:6]
z1_hz=torch.matmul(h0,gru.weight_hh_l0[3:6].T)+gru.bias_hh_l0[3:6]
z1=sigmoid(z1_iz+z1_hz)
# 第1时间步的候选隐藏状态
n1_in=torch.matmul(x[0],gru.weight_ih_l0[6:9].T)+gru.bias_ih_l0[6:9]
n1_hn=r1*(torch.matmul(h0,gru.weight_hh_l0[6:9].T)+gru.bias_hh_l0[6:9])
n1=tanh(n1_in+n1_hn)
# 第1时间步的隐藏状态
h1=(1-z1)*n1+z1*h0
# 第2时间步的重置门
r2_ir=torch.matmul(x[1],gru.weight_ih_l0[0:3].T)+gru.bias_ih_l0[0:3]
r2_hr=torch.matmul(h1,gru.weight_hh_l0[0:3].T)+gru.bias_hh_l0[0:3]
r2=sigmoid(r2_ir+r2_hr)
# 第2时间步的更新门
z2_iz=torch.matmul(x[1],gru.weight_ih_l0[3:6].T)+gru.bias_ih_l0[3:6]
z2_hz=torch.matmul(h1,gru.weight_hh_l0[3:6].T)+gru.bias_hh_l0[3:6]
z2=sigmoid(z2_iz+z2_hz)
# 第2时间步的候选隐藏状态
n2_in=torch.matmul(x[1],gru.weight_ih_l0[6:9].T)+gru.bias_ih_l0[6:9]
n2_hn=r2*(torch.matmul(h1,gru.weight_hh_l0[6:9].T)+gru.bias_hh_l0[6:9])
n2=tanh(n2_in+n2_hn)
# 第2时间步的隐藏状态
h2=(1-z2)*n2+z2*h1
# 输出
y=torch.cat((h1,h2),dim=0)
y,h2
```
## 2.14 长短期记忆(LSTM)
#### 2.14.1 LSTM的使用示例
```Python
# (4,3,1)的LSTM：输入特征数为4，隐藏单元个数为3，隐藏层数为1的LSTM
lstm=nn.LSTM(4,3,1)
# (2,8,4)的输入数据x：时间步长为2，批量大小为8，输入特征数为4的输入数据
# (1,8,3)的初始隐藏状态h0：隐藏层数为1，批量大小为8，隐藏单元个数为3的初始隐藏状态
# (1,8,3)的初始记忆细胞c0：隐藏层数为1，批量大小为8，隐藏单元个数为3的初始记忆细胞
x=torch.randn(2,8,4)
h0=torch.randn(1,8,3)
c0=torch.randn(1,8,3)
# (2,8,3)的输出数据y：时间步长为2，批量大小为8，隐藏单元个数为3的输出数据
# (1,8,3)的最终隐藏状态h2：隐藏层数为1，批量大小为8，隐藏单元个数为3的最终隐藏状态（即第2时间步的隐藏状态）
# (1,8,3)的最终记忆细胞c2：隐藏层数为1，批量大小为8，隐藏单元个数为3的最终记忆细胞（即第2时间步的记忆细胞）
y,(h2,c2)=lstm(x,(h0,c0))
y,(h2,c2)
```
#### 2.14.2 LSTM的前向传播过程
```Python
# LSTM的前向传播计算过程
sigmoid=nn.Sigmoid()
tanh=nn.Tanh()
# 第1时间步的输入门
i1_ii=torch.matmul(x[0],lstm.weight_ih_l0[0:3].T)+lstm.bias_ih_l0[0:3]
i1_hi=torch.matmul(h0,lstm.weight_hh_l0[0:3].T)+lstm.bias_hh_l0[0:3]
i1=sigmoid(i1_ii+i1_hi)
# 第1时间步的遗忘门
f1_if=torch.matmul(x[0],lstm.weight_ih_l0[3:6].T)+lstm.bias_ih_l0[3:6]
f1_hf=torch.matmul(h0,lstm.weight_hh_l0[3:6].T)+lstm.bias_hh_l0[3:6]
f1=sigmoid(f1_if+f1_hf)
# 第1时间步的候选记忆细胞
g1_ig=torch.matmul(x[0],lstm.weight_ih_l0[6:9].T)+lstm.bias_ih_l0[6:9]
g1_hg=torch.matmul(h0,lstm.weight_hh_l0[6:9].T)+lstm.bias_hh_l0[6:9]
g1=tanh(g1_ig+g1_hg)
# 第1时间步的输出门
o1_io=torch.matmul(x[0],lstm.weight_ih_l0[9:12].T)+lstm.bias_ih_l0[9:12]
o1_ho=torch.matmul(h0,lstm.weight_hh_l0[9:12].T)+lstm.bias_hh_l0[9:12]
o1=sigmoid(o1_io+o1_ho)
# 第1时间步的记忆细胞
c1=f1*c0+i1*g1
# 第1时间步的隐藏状态
h1=o1*tanh(c1)
# 第2时间步的输入门
i2_ii=torch.matmul(x[1],lstm.weight_ih_l0[0:3].T)+lstm.bias_ih_l0[0:3]
i2_hi=torch.matmul(h1,lstm.weight_hh_l0[0:3].T)+lstm.bias_hh_l0[0:3]
i2=sigmoid(i2_ii+i2_hi)
# 第2时间步的遗忘门
f2_if=torch.matmul(x[1],lstm.weight_ih_l0[3:6].T)+lstm.bias_ih_l0[3:6]
f2_hf=torch.matmul(h1,lstm.weight_hh_l0[3:6].T)+lstm.bias_hh_l0[3:6]
f2=sigmoid(f2_if+f2_hf)
# 第2时间步的候选记忆细胞
g2_ig=torch.matmul(x[1],lstm.weight_ih_l0[6:9].T)+lstm.bias_ih_l0[6:9]
g2_hg=torch.matmul(h1,lstm.weight_hh_l0[6:9].T)+lstm.bias_hh_l0[6:9]
g2=tanh(g2_ig+g2_hg)
# 第2时间步的输出门
o2_io=torch.matmul(x[1],lstm.weight_ih_l0[9:12].T)+lstm.bias_ih_l0[9:12]
o2_ho=torch.matmul(h1,lstm.weight_hh_l0[9:12].T)+lstm.bias_hh_l0[9:12]
o2=sigmoid(o2_io+o2_ho)
# 第2时间步的记忆细胞
c2=f2*c1+i2*g2
# 第2时间步的隐藏状态
h2=o2*tanh(c2)
# 输出
y=torch.cat((h1,h2),dim=0)
y,(h2,c2)
```
## 2.15 深度循环神经网络
#### 2.15.1 深度循环神经网络的使用示例
```Python
# (3,4,2)的循环神经网络：输入特征数为3，隐藏单元个数为4，隐藏层数为2的循环神经网络
rnn=torch.nn.RNN(3,4,2)
# (2,8,3)的输入数据x：时间步数为2，批量大小为8，特征数为3的输入数据，本质上是一个三维张量
# (2,8,4)的初始隐藏状态h0：隐藏层数为2，批量大小为8，隐藏单元个数为4的初始隐藏状态，本质上是一个三维张量
x=torch.randn(2,8,3)
h0=torch.randn(2,8,4)
# (2,8,4)的输出数据y：时间步数为2，批量大小为8，隐藏单元个数为4的输出数据，本质上是一个三维张量
# (2,8,4)的最终隐藏状态h2：隐藏层数为2，批量大小为8，隐藏单元个数为4的最终隐藏状态（即第2时间步的隐藏状态），本质上是一个三维张量
y,h2=rnn(x,h0)
y,h2
```
#### 2.15.2 深度循环神经网络的前向传播
```Python
# 深度循环神经网络的前向传播过程
tanh=nn.Tanh()
# 第1时间步
# 第1个时间步第1个隐藏层对应的初始隐藏状态
h1_ih1=torch.matmul(x[0],rnn.weight_ih_l0.T)+rnn.bias_ih_l0
h1_hh1=torch.matmul(h0[0],rnn.weight_hh_l0.T)+rnn.bias_hh_l0
h11=tanh(h1_ih1+h1_hh1)
# 第1个时间步第2个隐藏层对应的隐藏状态
h1_ih2=torch.matmul(h11,rnn.weight_ih_l1.T)+rnn.bias_ih_l1
h1_hh2=torch.matmul(h0[1],rnn.weight_hh_l1.T)+rnn.bias_hh_l1
h12=tanh(h1_ih2+h1_hh2)
h1=torch.cat((h11,h12)).view(2,h11.shape[0],h11.shape[1])
# 第2时间步
# 第2时间步第1个隐藏层对应的初始隐藏状态
h2_ih1=torch.matmul(x[1],rnn.weight_ih_l0.T)+rnn.bias_ih_l0
h2_hh1=torch.matmul(h1[0],rnn.weight_hh_l0.T)+rnn.bias_hh_l0
h21=tanh(h2_ih1+h2_hh1)
# 第2个隐藏层对应的隐藏状态
h2_ih2=torch.matmul(h21,rnn.weight_ih_l1.T)+rnn.bias_ih_l1
h2_hh2=torch.matmul(h1[1],rnn.weight_hh_l1.T)+rnn.bias_hh_l1
h22=tanh(h2_ih2+h2_hh2)
h2=torch.cat((h21,h22)).view(2,h21.shape[0],h21.shape[1])
y=torch.cat((h1[1],h2[1])).view(2,h1[1].shape[0],h1[1].shape[1])
y,h2
```
## 2.16 双向循环神经网络
#### 2.16.1 双向循环神经网络的使用示例
```Python
# (4,3,1,bidirectional=True)的循环神经网络：输入特征数为4，隐藏单元个数为3，隐藏层数为1的双向循环神经网络
rnn=nn.RNN(4,3,1,bidirectional=True)
# (2,8,4)的输入数据x：时间步长为2，批量大小为8，输入特征数为4的输入数据
# (1,8,3)的正向初始隐藏状态hd0：隐藏层数为1，批量大小为8，隐藏单元个数为3的正向初始隐藏状态
# (1,8,3)的反向初始隐藏状态hr3：隐藏层数为1，批量大小为8，隐藏单元个数为3的反向初始隐藏状态
# (2,8,3)的hd0_hr3：hd0与hr3在第1维度上的连结
x=torch.randn(2,8,4)
hd0=torch.randn(1,8,3)
hr3=torch.randn(1,8,3)
hd0_hr3=torch.cat((hd0,hr3),dim=0)
# (2,8,6)的输出数据y：时间步长为2，批量大小为8，隐藏单元个数为6（正向3+反向3）的输出数据
#     y[0]=torch.cat((hd1,hr1),dim=2)：hd1与hr1在第3维度上的连结
#     y[1]=torch.cat((hd2,hr2),dim=2)：hd2与hr2在第3维度上的连结
# (2,8,3)的隐藏状态hd2_hr1：hd2与hr1的在第1维度上的连结
#     (1,8,3)的正向隐藏状态hd2：隐藏层数为1，批量大小为8，隐藏单元个数为3的正向最终隐藏状态（时间步为2的正向隐藏状态）
#     (1,8,3)的反向隐藏状态hr1：隐藏层数为1，批量大小为8，隐藏单元个数为3的反向最终隐藏状态（时间步为1的反向隐藏状态）
y,hd2_hr1=rnn(x,hd0_hr3)
y,hd2_hr1
```
#### 2.16.2 双向循环神经网络的前向传播
```Python
# 双向循环神经网络的前向传播过程
tanh=nn.Tanh()
# 正向传播
# 第1时间步的正向隐藏状态
hd1_ih=torch.matmul(x[0],rnn.weight_ih_l0.T)+rnn.bias_ih_l0
hd1_hh=torch.matmul(hd0,rnn.weight_hh_l0.T)+rnn.bias_hh_l0
hd1=tanh(hd1_ih+hd1_hh)
# 第2时间步的正向隐藏状态
hd2_ih=torch.matmul(x[1],rnn.weight_ih_l0.T)+rnn.bias_ih_l0
hd2_hh=torch.matmul(hd1,rnn.weight_hh_l0.T)+rnn.bias_hh_l0
hd2=tanh(hd2_ih+hd2_hh)
# 正向传播对应输出数据yd
yd=torch.cat((hd1,hd2),dim=0)
# 逆向传播
# 第2时间步的逆向隐藏状态
hr2_ih=torch.matmul(x[1],rnn.weight_ih_l0_reverse.T)+rnn.bias_ih_l0_reverse
hr2_hh=torch.matmul(hr3,rnn.weight_hh_l0_reverse.T)+rnn.bias_hh_l0_reverse
hr2=tanh(hr2_ih+hr2_hh)
# 第1时间步的逆向隐藏状态
hr1_ih=torch.matmul(x[0],rnn.weight_ih_l0_reverse.T)+rnn.bias_ih_l0_reverse
hr1_hh=torch.matmul(hr2,rnn.weight_hh_l0_reverse.T)+rnn.bias_hh_l0_reverse
hr1=tanh(hr1_ih+hr1_hh)
# 逆向传播对应的输出数据yr
yr=torch.cat((hr1,hr2),dim=0)
# 最终的输出数据y
y=torch.cat((yd,yr),dim=2)
# 最终的隐藏状态：hd2_hr1
hd2_hr1=torch.cat((hd2,hr1),dim=0)
y,hd2_hr1
```

# 3 定义损失函数
## 3.1 均方误差函数
```Python
# 以均方误差的均值作为损失函数
lossMean=nn.MSELoss(reduction="mean")
# 任意维度的张量x
# 与张量x同维度的张量y
x=torch.randn(3,4)
y=torch.randn(3,4)
# 计算x相对于y的损失，等价于
# lm=((x-y)**2).mean()
lm=lossMean(x,y)
```
```Python
# 以均方误差的和作为损失函数
lossSum=nn.MSELoss(reduction="sum")
# 任意维度的张量x
# 与张量x同维度的张量y
x=torch.randn(3,4,5)
y=torch.randn(3,4,5)
# 计算x相对于y的损失，等价于
# ls=((x-y)**2).sum()
ls=lossSum(x,y)
```
## 3.2 一个包括softmax运算和交叉熵损失计算的函数(CrossEntropyLoss)
```Python
# softmax运算和交叉熵损失计算的函数的平均作为损失函数
loss=nn.CrossEntropyLoss(reduction="mean")
# (3,5)的张量x：批量大小为3，种类个数为5
# (3)的张量y：批量大小为3
x=torch.randn(3,5)
y=torch.empty(3,dtype=torch.long).random_(5)
# 计算x相对于y的损失，等价于（批量大小为3）
# l0=-x[0][y[0]]+torch.log(x[0].exp().sum())
# l1=-x[1][y[1]]+torch.log(x[1].exp().sum())
# l2=-x[2][y[2]]+torch.log(x[2].exp().sum())
# l=(l0+l1+l2)/3
l=loss(x,y)
```
```Python
# softmax运算和交叉熵损失计算的函数的和作为损失函数
loss=nn.CrossEntropyLoss(reduction="sum")
# (3,5)的张量x：批量大小为3，种类个数为5
# (3)的张量y：批量大小为3
x=torch.randn(3,5)
y=torch.empty(3,dtype=torch.long).random_(5)
# 计算x相对于y的损失，等价于（批量大小为3）
# l0=-x[0][y[0]]+torch.log(x[0].exp().sum())
# l1=-x[1][y[1]]+torch.log(x[1].exp().sum())
# l2=-x[2][y[2]]+torch.log(x[2].exp().sum())
# l=l0+l1+l2
l=loss(x,y)
```
# 4 定义优化算法
## 4.1 小批量随机梯度下降法(SGD)
```Python
# 形状为(3,3)，元素值都为1的自变量x
x=torch.ones(3,3,requires_grad=True)
# (1)的标量y
y=(2*(x**2)).sum()
# 求梯度
y.backward()
# 小批量随机梯度下降法：参数列表为[x]，学习率为0.1
optim=torch.optim.SGD([x],lr=0.1)
# 对参数进行第一次迭代，等价于
# x.data=x.data-lr*x.grad
optim.step()
# 对参数进行第二次迭代，等价于
# x.data=x.data-lr*x.grad
optim.step()
```
## 4.2 动量法(SGD的另一种使用方式)
```Python
# (3,3)的自变量x
x=torch.ones(3,3,requires_grad=True)
# (1)的标量y
y=(2*(x**2)).sum()
# 求梯度
y.backward()
# 动量法：参数列表为[x]，学习率为0.1，动量超参数为0.2
optim=torch.optim.SGD([x],lr=0.1,momentum=0.2)
# 对参数x进行第一次迭代，等价于（v0=0）
# v1=momentum*v0+lr*x.grad
# x.data=x.data-v1
optim.step()
# 对参数x进行第二次迭代，等价于
# v2=momentum*v1+lr*x.grad
# x.data=x.data-v2
optim.step()
```
## 4.3 AdaGrad算法
```Python
# (3,3)的自变量x
x=torch.ones(3,3,requires_grad=True)
# (1)的标量y
y=(2*(x**2)).sum()
# 求梯度
y.backward()
# AdaGrad算法：参数列表为[x]，学习率为0.1
optim=torch.optim.Adagrad([x],lr=0.1)
# 对参数x进行第一次迭代，等价于（s0=0）
# s1=s0+x.grad*x.grad
# x.data=x.data-lr*(1/(sqrt(s1)+0.000001))*x.grad
optim.step()
# 对参数x进行第二次迭代，等价于
# s2=s1+x.grad*x.grad
# x.data=x.data-lr*(1/(sqrt(s2)+0.000001))*x.grad
optim.step()
```
## 4.4 RMSprop算法
```Python
# (3,3)的自变量x
x=torch.ones(3,3,requires_grad=True)
# (1)的标量y
y=(2*(x**2)).sum()
# 求梯度
y.backward()
# RMSProp算法：参数列表为[x]，学习率为0.1，指数加权参数为0.8
optim=torch.optim.RMSprop([x],lr=0.1,alpha=0.8)
# 对参数x进行第一次迭代，等价于（s0=0）
# s1=alpha*s0+(1-alpha)*x.grad*x.grad
# x.data=x.data-lr*(1/(sqrt(s1)+0.000001))*x.grad
optim.step()
# 对参数x进行第二次迭代，等价于
# s2=alpha*s1+(1-alpha)*x.grad*x.grad
# x.data=x.data-lr*(1/(sqrt(s2)+0.000001))*x.grad
optim.step()
```
## 4.5 AdaDelta算法
```Python
# (3,3)的变量x
x=torch.ones(3,3,requires_grad=True)
# (1)的标量y
y=(2*(x**2)).sum()
# 求梯度
y.backward()
# Adadelta算法：参数列表为[x]，指数加权参数为0.8
optim=torch.optim.Adadelta([x],rho=0.8)
# 对参数x进行第一次迭代，等价于（s0=0,delta0=0）
# s1=rho*s0+(1-rho)*x.grad*x.grad
# deltax=sqrt((delta0+0.000001)/(s1+0.000001))*x.grad
# x.data=x.data-deltax
# delta1=rho*delta0+(1-rho)*deltax*deltax
optim.step()
# 对参数x进行第二次迭代，等价于
# s2=rho*s1+(1-rho)*x.grad*x.grad
# deltax=sqrt((delta1+0.000001)/(s2+0.000001))*x.grad
# x.data=x.data-deltax
# delta2=rho*delta1+(1-rho)*deltax*deltax
optim.step()
```
## 4.6 Adam算法
```Python
# (3,3)的变量x
x=torch.ones(3,3,requires_grad=True)
# (1)的标量y
y=(2*(x**2)).sum()
# 求梯度
y.backward()
# Adam算法：参数列表为[x]，学习率为0.001，指数加权超参数1为0.9，指数加权超参数2为0.999
optim=torch.optim.Adam([x],lr=0.001,betas=(0.9,0.999))
# 对参数x进行第一次迭代，等价于（v0=0,s0=0）
# v1=(betas[0]*v0+(1-betas[0])*x.grad)
# 对v1进行偏差修正，有时修正有时不修正
# v1=v1/(1-betas[0]**1)
# s1=(betas[1]*s0+(1-betas[1]*x.grad*x.grad))
# 对s1进行偏差修正，有时修正有时不修正
# s1=s1/(1-betas[1]**1)
# x.data=x.data-(lr*v1)/(s1**(1/2)+0.00000001)
optim.step()
# 对参数x进行第二次迭代，等价于
# v2=(betas[0]*v1+(1-betas[0])*x.grad)
# 对v2进行偏差修正，有时修正有时不修正
# v2=v2/(1-betas[0]**2)
# s2=(betas[1]*s1+(1-betas[1]*x.grad*x.grad))
# 对s2进行偏差修正，有时修正有时不修正
# s2=s2/(1-betas[1]**2)
# x.data=x.data-(lr*v2)/(s2**(1/2)+0.00000001)
optim.step()
```

# 5 训练模型
```Python
# net：神经网络模型
# loss：损失函数
# optim：优化器
# data_iter：训练数据集小批量迭代器
# batch_size：批量大小
# num_epochs：模型训练迭代次数
def train(data_iter,net,loss,optim,batch_size,num_epochs):
    for i in range(num_epochs):
        for x, y in data_iter:
            # 获取神经网络预测值
            y_hat=net(x) 
            # 获取损失
            l=loss(y_hat,y)
            # 梯度清零
            optim.zero_grad()
            # 求梯度
            l.backward()
            # 迭代参数进行优化
            optim.step()
```
```Python
# 训练数据集小批量迭代器data_iter：具体问题具体分析
data_iter
# 神经网络模型采用(1024,10)的全连接层
net=torch.nn.Linear(1024,10)
# 以均方误差的均值作为损失函数
loss=torch.nn.MSELoss(reduction="mean")
# 小批量随机梯度下降法作为优化器：参数列表为模型参数列表，学习率为0.1
optim=torch.nn.SGD(net.paramers(),lr=0.1)
# 批量大小为10
batch_size=10
# 模型训练迭代次数为3
num_epochs=3
# 调用训练函数
train(data_iter,net,loss,optim,batch_size,num_epochs)
```

# 6 应用模型
* 具体问题，具体分析。