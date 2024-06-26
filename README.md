# Graph-less Collaborative Filtering论文复现

## 介绍
图神经网络（GNN）在协同过滤（CF）中提供了最先进的网络架构。然而，现有的基于GNN的方法在准确性和可扩展性方面存在局限性，这些局限性是由噪声放大的过度平滑效应和耗时的迭代图编码过程引起的。

为了解决上述问题，我们提出利用知识蒸馏的力量，将由先进但复杂的GNN教师模型学习到的知识蒸馏到轻量级的MLP学生模型，通过这种方式，模型推理的时间复杂性得以降低，同时避免了导致过度平滑的复杂图聚合。具体而言，我们采用了双层蒸馏方法，包括预测层蒸馏和嵌入层蒸馏。预测层蒸馏对齐用户-物品关系的预测，而嵌入层蒸馏则使用对比学习对齐学习到的嵌入。一个对比正则化项被添加到优化目标中，以学习更均匀的表示。


## 实验
我们从以下两个方面验证了 SimRec 框架的有效性：

1. **SimRec 框架的不同子模块如何对整体性能做出贡献**
通过每次消除一个不同子模块，根据消除后的实验结果来判断被消除模块的作用，从而分析出不同模块对 SimRec 模型的有效性。

2. **调整 SimRec 模型的某些重要超参数时，模型的性能会如何变化**
使用不同的预测蒸馏权重 $\lambda_1$、嵌入层权重 $\lambda_2$、正则化参数 $\lambda_3$，通过比较 Recall 值和 NDGC 值，分析模型性能会发生怎样的变化。

## 环境
推荐使用的运行环境如下所示：
* python=3.10.4
* torch=1.11.0
* numpy=1.22.3
* scipy=1.7.3

## 数据集

我们使用了两个数据集来评估 SimRec 模型：<i>Gowalla</i> 和 <i>Amazon</i>。需要注意的是，与我们之前的工作中使用的数据相比，在本次研究中，我们使用了更加稀疏的三个数据集版本，以增加推荐任务的难度。我们的评估遵循常见的隐式反馈范式。数据集被划分为训练集、验证集和测试集，比例为 70:5:25。
| Dataset | \# Users | \# Items | \# Interactions | Interaction Density |
|:-------:|:--------:|:--------:|:---------------:|:-------:|
|Gowalla|$25,557$|$19,747$|$294,983$|$5.9\times 10^{-4}$|
|Amazon |$76,469$|$83,761$|$966,680$|$1.5\times 10^{-4}$|


## 代码包内容
```
SimRec-main
│  README.md        
├─Datasets
│  ├─sparse_amazon    
│  └─sparse_gowalla   
│  └─sparse_yelp
│      
├─History
├─methods
│  └─SimRec
│      │  DataHandler.py
│      │  Main.py
│      │  Model.py
│      │  Params.py
│      │  pretrainTeacher.py
│      │  
│      ├─Utils
│      │  │  TimeLogger.py
│      │  └─ Utils.py  
│              
└─Models
        
```

### 一些重要参数
* `reg`: 权重衰减正则化的权重，我们可以从集合`{1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8}`中调整这个超参数.
* 正则化参数，推荐从以下集合中选择 `{10, 3, 1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3}`.
  * `softreg`: 预蒸馏阶段的权重。
  * `cdreg`:嵌入级蒸馏的权重。
  * `screg`: 对比正则化权重。
* 正则化参数的温度推荐表如下： `{3, 1, 3e-1, 1e-1, 3e-2, 1e-2}`.
  * `tempsoft`: 预蒸馏温度系数。
  * `tempcd`: 嵌入级蒸馏温度系数。
  * `tempsc`: 对比正则化温度系数。
* `teacher_model`: 用于使用已训练好的教师模型。
* `load_model`: 使用已训练好的模型。
* `save_path`: 选择保存模型的位置。

## 使用方法
在使用这个代码包时，你可以选择将文件夹History中的记录以及Model中的模型删除来得到自己的模型，若你选择删除，那么你需要在主文件夹SimRec-main下创建同样的两个文件夹来保存相关内容。

之后你需要进入'methods/SimRec/'文件夹内部来运行以下代码，以<i>Gowalla</i>数据集为例：


* Gowalla
```
python Main.py --data gowalla --sc 1 --soft 1e-1 --cd 1e-2 --sc 1 --save_path 你模型的名称 --teacher_model 所使用的教师模型名称
```

上述代码指的是你使用数据集<i>Gowalla</i>以及相关参数来运行代码Main.py，后面的save_path表示你想保存的模型名称，若不加这段则默认保存为tem名称，而teacher_model后接你想使用的预训练完成的教师模型。

为了模型效果可以达到最佳，你可以尝试修改不同参数后的值，通过观测模型效果来进一步调参。

```
python Main.py --data gowalla --sc 1 --soft 1e-1 --cd 1e-2 --sc 1 --load_model 想要使用的模型名称
```

通过这种方式就可以使用已训练完成的模型来运行代码，而本代码也支持对一个数据集一次运行多个模型，只要在load_model后依次填入多个模型名称即可。

```
python pretrainTeacher.py --data gowalla --sc 1 --soft 1e-1 --cd 1e-2 --sc 1 --save_path 你模型的名称
```
使用该代码可以得到相应数据的预训练模型。
