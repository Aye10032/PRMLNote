# 附 第六章作业

## 作业1

### 题目

考虑一种特定类别的高斯朴素贝叶斯分类器。其中：

- y是一个布尔变量，服从伯努利分布，参数为$\pi=P(y=1)$，因此$P(Y=1)=1-\pi$
- $x=\left[x_1,\dots,x_D\right]^T$，其中每个特征$x_i$是一个连续随机变量。对于每个$x_i$，$P(x_i\vert y=k)$是一个高斯分布$N(\mu_{ik},\sigma_i)$，其中$\sigma_i$是高斯分布的标准差，不依赖于$k$
- 对于所有$i\neq j$，给定$y$，$x_i$和$x_j$是条件独立的（即所谓“朴素”分类器）

问：证明上述这种高斯朴素贝叶斯判别器与逻辑回归得到的分类器形式是一致的



### 解

已知Logitstc回归的一般形式：
$$
P(Y=1\mid X)=\frac{1}{1+\exp(w_0+\sum\limits_{i=1}^nw_ix_i)}\\
\\
P(Y=0\mid X)=\frac{\exp(w_0+\sum\limits_{i=1}^nw_ix_i)}{1+\exp(w_0+\sum\limits_{i=1}^nw_ix_i)}
$$
根据题目中高斯朴素贝叶斯分类器的假设，有：
$$
\begin{align}
P(Y=1\mid X) & \frac{P(Y=1)P(X\mid Y=1)}{P(Y=1)P(X\mid Y=1)+P(Y=0)P(X\mid Y=0)}\\
\\
&=\frac1{1+\frac{P(Y=0)P(X\mid Y=0)}{P(Y=1)p(X \mid Y=1)}}\\
\\
&=\frac1{1+\exp\left(\ln\frac{P(Y=0)P(X\mid Y=0)}{P(Y=1)P(X \mid Y=1)}\right)}
\end{align}
$$
由给定的Y，x条件独立性假设，可得：
$$
\begin{align}
P(Y=1\mid X) &=\frac1{1+\exp\left(\ln\frac{P(Y=0)}{P(Y=1)}+\ln\left(\prod_i\frac{P(x_i\mid Y=0)}{P(x_i \mid Y=1)}\right)\right)}\\
\\
&=\frac1{1+\exp\left(\ln\frac{P(Y=0)}{P(Y=1)}+\sum_i\frac{P(x_i\mid Y=0)}{P(x_i \mid Y=1)}\right)}
\end{align}
$$




# 作业二

## 题目

- 去掉$p(x_i\vert y=k)$的标准差$\sigma_i$不依赖于k的假设，即对于每一个$x_i$，$P(x_i\vert y=k)$是一个高斯分布$N(\mu_{ik},\sigma_i)$，其中$i=1,\dots ,D$，$k=0$

问：这个更一般的高斯朴素贝叶斯分类器所隐含的$P(x\vert y)$的新形式仍然是逻辑回归所使用的形式吗？推导$P(x\vert y)$的新形式来证明你的答案



## 解

在更一般的高斯朴素贝叶斯分类器中，每个特征$x_i$的条件概率分布$P(x_i|y=k)$都是一个高斯分布$N(\mu_{ik}, \sigma_i)$，其中 $i=1, ..., D$，$k=0$。

则有：

$P(y=0|x) = \dfrac{P(x|y=0)P(y=0)}{P(x)} P(y=1|x) = \dfrac{P(x|y=1)P(y=1)}{P(x)}$

由于是二元分类，所以$P(y=1) = 1 - P(y=0)$。因此有：

$$
\dfrac{P(y=0|x)}{P(y=1|x)} = \dfrac{P(x|y=0)}{P(x|y=1)} \times \dfrac{1 - \pi}{\pi} \tag{1}
$$

其中，$\pi = P(y=1)$是类别为1的概率。

由于$P(x\vert y=k)$是一个高斯分布，则对于$P(x_i\vert y=0)$和$P(x_i\vert y=1)$分别记作$N(\mu_{0k}, \sigma_i)$和$N(\mu_{1k}, \sigma_i)$，进而有联合概率分布：
$$
\begin{align}
P(x_i|y=0) = P(x_1, ..., x_D|y=0) = \prod_{i=1}^{D} P(x_i|y=0) = \prod_{i=1}^{D} N(x_i|\mu_{0i}, \sigma_i) \tag{2}
\\
P(x_i|y=1) = P(x_1, ..., x_D|y=1) = \prod_{i=1}^{D} P(x_i|y=1) = \prod_{i=1}^{D} N(x_i|\mu_{1i}, \sigma_i) \tag{3}
\end{align}
$$


将等式 (2) 和 (3) 代入等式 (1) ，得：

$$
\frac{P(y=0|x)}{P(y=1|x)} = \frac{\prod\limits_{i=1}^{D} N(x_i|\mu_{0i}, \sigma_i)}{\prod\limits_{i=1}^{D} N(x_i|\mu_{1i}, \sigma_i)} \times \frac{1 - \pi}{\pi}
$$

等式两边同时取对数，得：

$$
\begin{align}
\log\left(\frac{P(y=0|x)}{P(y=1|x)}\right) &= \log\left(\frac{1 - \pi}{\pi}\right) + \sum_{i=1}^{D} \log\left(\frac{N(x_i|\mu_{0i}, \sigma_i)}{N(x_i|\mu_{1i}, \sigma_i)}\right)
\\
&= \log\left(\frac{1 - \pi}{\pi}\right) + \sum_{i=1}^{D} \left[\log\left(\frac{1}{\sqrt{2\pi}\sigma_i}\right) - \frac{(x_i-\mu_{0i})^2}{2\sigma_i^2} + \frac{(x_i-\mu_{1i})^2}{2\sigma_i^2}\right]
\\
&= \log\left(\frac{1 - \pi}{\pi}\right) + \sum_{i=1}^{D} \left[\frac{\mu_{1i}-\mu_{0i}}{\sigma_i^2}x_i - \frac{\mu_{1i}^2-\mu_{0i}^2}{2\sigma_i^2}\right] + C
\end{align}
$$

其中C是与特征$x_i$无关的常数。继续整理等式，将其变为：
$$
\log\left(\frac{P(y=0|x)}{P(y=1|x)}\right) = \sum_{i=1}^{D} \left[\frac{\mu_{1i}-\mu_{0i}}{\sigma_i^2}x_i - \frac{\mu_{1i}^2-\mu_{0i}^2}{2\sigma_i^2}\right] + C'
$$

其中$C^{\\'}$是另一个与特征$x_i$无关的常数。现在，我们可以将对数比率和特征的线性组合形式结合起来：

$$
\log\left(\frac{P(y=0|x)}{P(y=1|x)}\right) = \mathbf{w}^T \mathbf{x} + b
$$

这里，$\mathbf{w}=\left[\frac{\mu_{10}-\mu_{00}}{\sigma_1^2}, \frac{\mu_{11}-\mu_{01}}{\sigma_2^2}, ..., \frac{\mu_{1D}-\mu_{0D}}{\sigma_D^2}\right]$是一个参数向量，$b=\sum\limits_{i=1}^{D} \left[\frac{\mu_{1i}^2-\mu_{0i}^2}{2\sigma_i^2}\right] - \log\left(\frac{1-\pi}{\pi}\right)$是一个常数项。

因此，通过上述推导，$P(x|y)$的新形式为：


$$
P(x|y) = \frac{1}{1+\exp(-(\mathbf{w}^T \mathbf{x} + b))}
$$

这正是逻辑回归模型中使用的形式。因此，更一般的高斯朴素贝叶斯分类器所隐含的P(x|y)的新形式与逻辑回归所使用的形式是一致的。



# 作业三

## 题目

现在，考虑我们的高斯贝叶斯分类器的以下假设（不是“朴素”的）：

- $y$是符合伯努利分布的布尔变量，参数$\pi=P(y=1)$，$P(y=0)=1-\pi$
- $x=\left[x_1,x_2\right]^T$，即每个样本只考虑两个特征，每个特征为连续随机变量， 假设$P(x_1,x_2\vert y=k)$是一个二元高斯分布$N(\mu_{1k},\mu_{2k},\sigma_1, \sigma_2,\rho)$，其中$\mu_{1k}$和$\mu_{2k}$是$x_1$和$x_2$的均值，$\sigma_1$和$\sigma_2$是$x_1$和$x_2$的标准差，$\rho$是$x_1$和$x_2$的相关性。二元高斯分布的概率密度为：

$$
P\left(x_1, x_2 \mid y=k\right)=\frac{1}{2 \pi \sigma_1 \sigma_2 \sqrt{1-\rho^2}} \exp \left[-\frac{\sigma_2^2\left(x_1-\mu_{1 k}\right)^2+\sigma_1^2\left(x_2-\mu_{2 k}\right)^2-2 \rho \sigma_1 \sigma_2\left(x_1-\mu_{1 k}\right)\left(x_2-\mu_{2 k}\right)}{2\left(1-\rho^2\right) \sigma_1^2 \sigma_2^2}\right]
$$

问：这种不那么朴素的高斯贝叶斯分类器所隐含的$P(x\vert y)$的形式仍然是逻辑回归所使用的形式吗？推导$P(y\vert x)$的形式来证明你的答案



## 解

根据贝叶斯定理，有：

$$
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
$$

首先计算$P(x|y)$的概率密度函数：

$$
P(x|y=k) = \frac{1}{2 \pi \sigma_1 \sigma_2 \sqrt{1-\rho^2}} \exp \left[-\frac{\sigma_2^2\left(x_1-\mu_{1k}\right)^2+\sigma_1^2\left(x_2-\mu_{2k}\right)^2-2 \rho \sigma_1 \sigma_2\left(x_1-\mu_{1k}\right)\left(x_2-\mu_{2k}\right)}{2\left(1-\rho^2\right) \sigma_1^2 \sigma_2^2}\right]
$$

然后计算P(x)的边缘概率密度函数。由于只有两个特征，我们可以把它们视为多元高斯分布的条件下界的一部分。因此，可以计算P(x)如下：

$$
P(x) = \sum_k P(x|y=k)P(y=k) = P(x|y=1)P(y=1) + P(x|y=0)P(y=0)
$$

接下来，计算$P(y=1|x)$和$P(y=0|x)$：

$$
\begin{align}
P(y=1|x) = \frac{P(x|y=1)P(y=1)}{P(x)} = \frac{P(x|y=1)P(y=1)}{P(x|y=1)P(y=1) + P(x|y=0)P(y=0)}
\\
P(y=0|x) = \frac{P(x|y=0)P(y=0)}{P(x)} = \frac{P(x|y=0)P(y=0)}{P(x|y=1)P(y=1) + P(x|y=0)P(y=0)}
\end{align}
$$

将P(x|y)的表达式代入上面的公式，并利用比例关系简化公式：

$$
P(y=1|x) = \frac{1}{1 + \frac{P(x|y=0)P(y=0)}{P(x|y=1)P(y=1)} \cdot \frac{P(y=1)}{P(y=0)}}
$$

注意到$P(y=1)/P(y=0)$是一个常数，可以化简上式：

$$
P(y=1|x) = \frac{1}{1 + \frac{P(x|y=0)P(y=0)}{P(x|y=1)P(y=1)} \cdot \frac{P(y=1)}{P(y=0)}} = \frac{1}{1 + \frac{P(x|y=0)P(y=0)}{P(x|y=1)P(y=1)} \cdot \frac{1-\pi}{\pi}}
$$

将$P(x|y=0)$和$P(x|y=1)$的表达式代入上式，并进行计算和化简。最终，我们会得到一个与逻辑回归形式相似的表达式，但其中的权重项**会受到先验概率和$P(x)$的影响**。因此，这种不那么朴素的高斯贝叶斯分类器所隐含的$P(x|y)$的形式不同于逻辑回归所使用的形式。



# 作业四

## 题目

利用表格中的数据训练朴素贝叶斯分类器
$$
\begin{align}
x_1 &= \{1,2,3\}
\\
x_2 &= \{S,M,L,N\}
\\
y &= \{1,-1\}
\end{align}
$$

|      |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |  10  |  11  |  12  |  13  |  14  |  15  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  x1  |  1   |  1   |  1   |  1   |  1   |  2   |  2   |  2   |  2   |  2   |  3   |  3   |  3   |  3   |  3   |
|  x2  |  S   |  M   |  M   |  S   |  S   |  S   |  M   |  M   |  L   |  L   |  L   |  M   |  M   |  L   |  L   |
|  y   |  -1  |  -1  |  1   |  1   |  -1  |  -1  |  -1  |  1   |  1   |  1   |  1   |  1   |  1   |  1   |  -1  |

给定测试样本$x=(2,S)^T$和$x=(1,N)^T$，请预测他们的标签



## 解

```python
from sklearn.naive_bayes import GaussianNB
import numpy as np

label = {'S': 0, 'M': 1, 'L': 2, 'N': 3}

x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
x2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

X = np.array([x1, [label.get(x, -1) for x in x2]]).T

clf = GaussianNB()

clf.fit(X, y)

test_samples = [[2, label['S']], [1, label['N']]]
predictions = clf.predict(test_samples)

for index, predict in enumerate(predictions):
    print(f'sample{index}: {predict}')
```

sample0: -1
sample1: 1

