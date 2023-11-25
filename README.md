# Triple Adversarial Learning for Influence based Poisoning Attack in Recommender Systems
This project is for the paper: [Triple Adversarial Learning for Influence based Poisoning Attack in Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3447548.3467335), Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021: 1830-1840.

The code was developed on Python 3.6 and tensorflow 1.14.0.

## Usage

### run generate_fake.py
```
usage: python generate_fake.py [--dataset DATA_NAME] [--gpu GPU_ID]
[--epochs EPOCHS] [--data_size DATA_SIZE] [--target_index TARGET_ITEMS]

optional arguments:
  --dataset DATA_NAME
                        Supported: filmtrust, ml-100k, ml-1m.
  --gpu GPU_ID
                        GPU ID, default is 0.
  --epochs EPOCHS
                        Training epochs.
  --data_size DATA_SIZE
                        The data available to the attacker.
  --target_index TARGET_ITEMS
                        The index of predefined target item list: 0, 1 for ml-100k, 2,3 for ml-1m, 4,5 for filmtrust, 6,7 for yelp.
```

### Example.
```bash
python generate_fake.py --dataset ml-100k --gpu 0 --target_index 0
```
提供的代码片段是用于对推荐系统实施对抗性攻击的复杂系统的一部分，特别是使用“推荐系统中基于影响力的中毒攻击的三重对抗性学习”方法。我将总结主要文件的关键组件和功能：

### 1.evaluate.py
功能：该文件作为评估推荐系统在攻击前后性能的主要入口点。
关键部件：
它设置 TensorFlow 环境并加载数据集。
初始化推荐系统模型（SVD）。
训练推荐系统并使用来自的实用函数评估其性能utils.py。
使用执行对抗性攻击utils.attack并在攻击后重新评估系统。
### 2.generate_fake.py
功能：专注于生成虚假用户数据来毒害推荐系统。
关键部件：
加载数据集并设置 GAN（生成对抗网络）模型TrialAttack.py。
训练 GAN 模型来生成虚假用户数据。
使用生成的数据执行投毒攻击。
### 3.greedy_selection.py
功能：实现选择攻击用户的逻辑。
关键部件：
使用 GAN 模型的输出来确定要定位的用户。
根据 GAN 生成的影响力分数对用户进行分组并做出选择。
修改数据集以包含对抗性示例。
### 4.k_means.py
功能：提供 k-means 聚类实用程序。
关键部件：
用于计算簇的平均点、更新簇中心以及将点分配给簇的函数。
该k_means函数协调聚类过程。
### 5.TrailAttack.py（GAN类）
功能：对抗性攻击的核心，实现用于生成虚假用户数据的 GAN。
关键部件：
为生成器和鉴别器定义占位符、变量和神经网络层。
实现 GAN 的训练例程，包括损失计算和优化。
包含计算用户影响力分数的方法，这对于攻击策略至关重要。
保存并加载模型以生成对抗性数据。
### 6.utils.py
功能：包含数据处理和模型评估的实用函数。
关键部件：
sampling以及get_batchs准备训练数据的函数。
recommend评估推荐系统的函数。
estimate_dataset用于使用生成的数据调整数据集的函数。
用于高级模型分析的 Hessian 向量和扰动向量乘积函数。

总之，这些文件共同形成一个系统，用于训练 GAN 生成对抗性用户数据，然后使用该数据毒害推荐系统并评估攻击的影响。该系统采用k-means聚类、影响力得分计算和对抗训练等先进技术来有效执行和评估中毒攻击。


### SVD.py
SVD 模型作为推荐系统的基础模型，然后进行攻击。evaluate.py 文件的流程大致为初始化 SVD 模型，对其进行训练和评估，然后实施投毒攻击并再次评估模型性能以量化攻击的影响。通过这种方式，可以观察到攻击前后推荐系统性能的变化，从而了解攻击的有效性。

