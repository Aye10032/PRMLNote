# 附 第七章作业

## 作业1

### 题目

给定如下训练数据集
$$
x_1 = [3\ 3]\\ 
y^1 = 1\\
x_2 = [4\ 3]\\
y^2 = 1\\
x_3 = [1\ 1]\\
y^3 = -1
$$
通过求解SVM的对偶问题来求解最大间隔的分类超平面    



### 解

首先，对偶问题为：
$$
\begin{align}
\max\limits_{\alpha} &\sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j=1}^N y^i y^j\alpha_i\alpha_j (\mathbf{x}^i)^T\mathbf{x}^j
\\
\operatorname{s.t.}&\ \alpha_i\geq0,i=1,\dots,N
\\
&\sum_{i=1}^N \alpha_i y^i = 0
\end{align}
$$
并求的其最优解$\boldsymbol{\alpha}^*=(\alpha_1^*,\dots,\alpha_l^*)$

则得到原问题的最优解：
$$
\begin{align}
&\mathbf{w}^*=\sum_{i=1}^N\alpha_i^*y^i\mathbf{x}^i
\\
&b^* = y^j-\sum_{i=1}^N\alpha_i^*y^i(\mathbf{x}^i)^T\mathbf{x}^j
\\
& \alpha_j^* > 0
\end{align}
$$
进而可以得到分离超平面：
$$
(\mathbf{w}^*)^T\mathbf{x} + b^* = 0
$$



对于给定的训练数据集，有：

$$
\begin{align} 
\mathbf{x}^1 &= [3\ 3] \\ 
\mathbf{x}^2 &= [4\ 3] \\ 
\mathbf{x}^3 &= [1\ 1] \\ 
y^1 &= 1 \\ y^2 &= 1 \\ y^3 &= -1 
\end{align}
$$

则有：
$$
k = \mathbf x^T \cdot \mathbf x = \begin{pmatrix}
18&21&6\\
21&25&7\\
6&7&2
\end{pmatrix}
$$



目标函数为：
$$
\begin{align} 
\max\limits_{\alpha} &\sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j=1}^N y^i y^j\alpha_i\alpha_j k_{ij} 
\end{align}
$$

约束条件为：

$$
\begin{align} 
\alpha_i \geq 0, i=1,\dots,N \\ \sum_{i=1}^N \alpha_i y^i = 0 
\end{align}
$$



### 代码实现

#### 使用scipy求解最优化

```python
import numpy as np
from scipy.optimize import minimize

X = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])
k = np.dot(X, X.T)


def objective(alpha):
    return -np.sum(alpha) + 0.5 * np.sum(np.outer(y, y) * np.outer(alpha, alpha) * k)


def constraint1(alpha):
    return alpha


def constraint2(alpha):
    return np.dot(alpha, y)


N = X.shape[0]
bounds = [(0, None)] * N
constraints = [{'type': 'ineq', 'fun': constraint1}, {'type': 'eq', 'fun': constraint2}]

alpha_initial = np.zeros(N)
result = minimize(objective, alpha_initial, bounds=bounds, constraints=constraints)

alphas = result.x
print(f'Optimal alphas:[{alphas[0]},{alphas[1]},{alphas[2]}]')

w = np.dot(alphas * y, X)
print(w)

b_index = np.argmax(alphas)
b = y[b_index] - np.sum(alphas * y * np.dot(X, X[b_index]))
print(b)
```

输出结果为：

w=[0.49999996 0.49999996]
b=-1.999999927629584



#### 使用sklearn中的SVM模块求解

```python
import numpy as np
from sklearn.svm import SVC

X = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])

svm = SVC(kernel='linear')
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]

print(w)
print(b)
```

输出结果为：

w = [0.5 0.5]
b = -2.0



## 作业2

### 题目

高斯核有以下形式：
$$
K(x,z) = \exp \left(-\frac{\Vert x-z \Vert^2}{2\sigma^2}\right)
$$
请证明高斯核函数可以表示为一个无穷维特征向量的内积

**提示**：利用以下展开式，将中间的因子展开为幂级数
$$
K(x,z) = \exp \left(-\frac{x^Tx}{2\sigma^2}\right) \exp \left(\frac{x^Tz}{2\sigma^2}\right) \exp \left(-\frac{z^Tz}{2\sigma^2}\right)
$$


### 证明

首先，有：
$$
\begin{align}
K(x,z) &= \exp \left(-\frac{\Vert x-z \Vert^2}{2\sigma^2}\right)
\\
&= \exp \left(-\frac{x^2+z^2-2xz}{2\sigma^2}\right)
\\
&= \exp \left(-\frac{x^2+z^2}{2\sigma^2}\right) \exp \left(\frac{xz}{\sigma^2}\right) \tag{1}
\end{align}
$$
由于函数$$e^x$$的幂级数展开式为：
$$
\begin{align}
e^x &= \sum_{i=0}^\infin\frac{x^n}{n!}\\
&= 1 + x + \frac{x^2}{2!} + \dots + \frac{x^n}{n!} + R_n
\end{align}
$$
因此，可以有：
$$
\begin{align}
\exp \left(\frac{xz}{\sigma^2}\right) &= 1 + \left(\frac{xz}{\sigma^2}\right) + \frac{\left(\frac{xz}{\sigma^2}\right)^2}{2!} + \dots + \frac{(\frac{xz}{\sigma^2})^n}{n!} + \dots
\\
&=1 + \frac1{\sigma^2}\cdot\frac{xz}{1!} + \left(\frac1{\sigma^2}\right)^2\cdot\frac{(xz)^2}{2!} + \dots + \left(\frac1{\sigma^2}\right)^n\cdot\frac{(xz)^n}{n!} + \dots
\\
&= 1\cdot1 + \frac1{1!}\frac x\sigma\cdot\frac z\sigma + \frac1{2!}\frac{x^2}{\sigma^2}\cdot\frac{z^2}{\sigma^2} +\dots+\frac1{n!}\frac{x^n}{\sigma^n}\cdot\frac{z^n}{\sigma^n} + \dots
\end{align}
$$
将其带回（1）式，有：
$$
\begin{align}
K(x,z) &= \exp \left(-\frac{x^2+z^2}{2\sigma^2}\right)\cdot\left(1\cdot1 + \frac1{1!}\frac x\sigma\cdot\frac z\sigma + \frac1{2!}\frac{x^2}{\sigma^2}\cdot\frac{z^2}{\sigma^2} +\dots+\frac1{n!}\frac{x^n}{\sigma^n}\cdot\frac{z^n}{\sigma^n} + \dots\right)
\\
&= \exp \left(-\frac{x^2}{2\sigma^2}\right)\cdot\exp \left(-\frac{z^2}{2\sigma^2}\right)\cdot\left(1\cdot1 + \frac1{1!}\frac x\sigma\cdot\frac z\sigma + \frac1{2!}\frac{x^2}{\sigma^2}\cdot\frac{z^2}{\sigma^2} +\dots+\frac1{n!}\frac{x^n}{\sigma^n}\cdot\frac{z^n}{\sigma^n} + \dots\right)
\\
&= \Phi(x)^T\cdot\Phi(z)
\end{align}
$$
其中，
$$
\Phi(x) = \exp \left(-\frac{x^2}{2\sigma^2}\right)\cdot\left(1 , \sqrt\frac1{1!}\frac x\sigma , \sqrt\frac1{2!}\frac{x^2}{\sigma^2} , \dots , \sqrt\frac1{n!}\frac{x^n}{\sigma^n} , \dots\right)
$$
因此，高斯核函数可以表示为一个无穷维特征向量的内积
