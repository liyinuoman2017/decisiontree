# decisiontree
从零入手人工智能（5）—— 决策树
## 1.前言

上视频我们讲解了逻辑回归这个分类算法，今天我们分享是决策树。决策树和逻辑回归这两种算法都属于分类算法，**以下是它们的相同点**：

**分类任务**：两者都是用于分类任务的算法。无论是决策树还是逻辑回归，它们的目标都是根据输入的特征（或变量）来预测样本的类别。

**预测类别**：它们都可以预测样本属于哪个类别。无论是二分类问题还是多分类问题，决策树和逻辑回归都能够进行建模和预测。

**处理特征**：两者都可以处理多种类型的特征，包括数值型特征和类别型特征。

**模型评估**：两者都可以使用相同的评估指标来评估模型的性能，如准确率、召回率、F1分数等。

![](file://C:\Users\Administrator\AppData\Roaming\marktext\images\2024-07-21-22-09-57-image.png)

虽然决策树和逻辑回归有上述相同点，但它在仍然存在差异。决**策树和逻辑回归最大的差异在于它们的模型算法原理不同**：**决策树**基于树形结构进行决策，通过一系列规则对数据进行分类。而**逻辑回归**使用逻辑函数（如sigmoid函数）对输入特征进行建模，将线性模型的输出转换为概率值，然后根据概率值判断样本所属的类别。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/10948b6b2d7d42f596d59f7b8af399a6.png)

我们直接通过一个入门程序和一个进阶实战来熟悉决策树。

## 2.入门程序

入门程序自动生成一组30个样品的特征数据，每个样品包含3个特征，30个样品对应了苹果、香蕉、橙子这三种水果。使用DecisionTreeClassifier方法建立一个决策树模型，训练模型，最后使用 plot_tree 函数可视化决策树的结构。


![](file://C:\Users\Administrator\AppData\Roaming\marktext\images\2024-07-21-21-32-49-image.png)


```c
import numpy as np  
from sklearn.tree import DecisionTreeClassifier, plot_tree  
import matplotlib.pyplot as plt  
import matplotlib  

# 设置随机种子以确保结果可复现  
np.random.seed(42)   
# 模拟数据集  
# 设置3个特征：
#第一个特征: 颜色（0=红色，1=黄色，2=橙色），
#第二个特征:形状（0=圆形，1=长形），
#第三个特征:甜度（0=不甜，1=甜）  
num_samples = 30  # 设置30个样品数
features = np.random.randint(0, 3, size=(num_samples, 3))  # 生成随机数据， 数据范围0~3，数据格式为 [30，3 ]

# 添加一些噪声和偏置，以确保数据不是完全线性的  
features[:, 0] = (features[:, 0] + np.random.normal(0, 0.2, num_samples)).round()  

# 设置苹果、香蕉、橙子这三种水果的特征关系  
targets = np.zeros(num_samples) 
# 红色圆形 -> 苹果      targets = 0 
targets[(features[:, 0] == 0) & (features[:, 1] == 0)] = 0 
# 黄色甜 -> 香蕉   targets = 1 
targets[(features[:, 0] == 1) & (features[:, 2] == 1)] = 1 
 # 橙色或长形不甜 -> 橙子  targets = 2
targets[(features[:, 0] == 2) | (features[:, 1] == 1) & (features[:, 2] == 0)] = 2   
```

```c
# 创建决策树模型  
clf = DecisionTreeClassifier(random_state=42)  
clf.fit(features, targets)  

# 可视化决策树  
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体（根据你的系统可能需要更改）  
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题  

plt.figure(figsize=(20, 15))  
plot_tree(clf, filled=True, feature_names=['颜色', '形状', '甜度'], class_names=['苹果', '香蕉', '橙子'])  
plt.show()
```

`gini`表示该节点的基尼不纯度，`samples`表示该节点包含的样本数量，`value`表示该节点中每个类别的样本数量。

![](file://C:\Users\Administrator\AppData\Roaming\marktext\images\2024-07-21-21-21-56-image.png)


## 3.进阶实战

本实战程序的目的是：利用气象环境数据表macau_weather.csv中的数据进行训练决策树模型，然后根据气象环境数据预测是否会下雨。
#### step1
读取macau_weather.csv中的数据，并可视化数据，根据可视化结果可知数据表中有以下数：

> num、date、air_pressure、high_tem、aver_tem、low_tem、	humidity、sunlight_time	、wind_direction、wind_speed、rain_accum

其中rain_accum为目标值（标签：有雨、无雨），以下七个数据为特征变量：

> air_pressure、high_tem、aver_tem、low_tem	、humidity、sunlight_time	、wind_direction、wind_speed


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/3337e1bd1bef464e9a9bba1bde5405d3.png)


#### step2
数据表中的一共有426组数据（来源于426天的气象数据记录），检查每组数据是否完整，根据检查结果可知有0.7%的数据存在空缺

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/9afd387e64674df3a6860ab0c09c27ec.png)

#### step3
将数据表中的rain_accum转换成1和0，0代表无雨1代表有雨。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ac1b29e1342742eba1bdeee82f0fa999.png)

#### step4
使用DecisionTreeClassifier方法建立决策树模型，利用训练集数据训练模型。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/ac1e0eaae3c947abb30ea2cc985a0557.png)

#### step5
利用模型和测试集数据，测试模型准确性，并可视化结果，根据可视化图标可知模型预测的准确性达到了87.1%。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/438b2a73c6a246e5b90aa26b8b5675ca.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/addf5635df59455d975e60b52e4db083.png)

**希望获取源码和测试数据的朋友请在评论区留言**

**创作不易希望朋友们点赞，转发，评论，关注!

您的点赞，转发，评论，关注将是我持续更新的动力!**
