# 附 第四章作业

## Q1

### 题目

设有如下三类模式样本集$$\omega_1$$，$$\omega_2$$和$$\omega_3$$，其先验概率相等，求$$S_w$$和$$S_b$$
$$
\begin{align}
&\omega_1: \{(1\ 0)^T,(2\ 0)^T,(1\ 1)^T\}\\
&\omega_2: \{(-1\ 0)^T,(0\ 1)^T,(-1\ 1)^T\}\\
&\omega_3: \{(-1\ -1)^T,(0\ -1)^T,(0\ -2)^T\}
\end{align}
$$

### 解

由题意可知
$$
P(\omega_1)=P(\omega_2)=P(\omega_3) = \frac{1}{3}
$$
先算出样本均值：
$$
m_1=\left(\frac{4}{3}\ \frac{1}{3}\right)^T
\\
m_2=\left(-\frac{2}{3}\ \frac{2}{3}\right)^T
\\
m_3=(-\frac{1}{3}\ -\frac{4}{3})^T
$$
则可得协方差矩阵：
$$
C_i = \frac{1}{N}\sum_{j=1}^{N_i}(x_{ij}-m_i)(x_{ij}-m_i)^T
$$
进而得到类内离散度矩阵：
$$
S_w = \sum_{i=1}^c P(\omega_i)C_i
$$
而对于类间离散度矩阵，直接用均值即可：
$$
\sum_{i=1}^cP(\omega_i)(m_i-m_0)(m_i-m_0)^T
$$
具体计算我这里通过numpy计算得到：

```python
import numpy as np

w1 = np.array([[1, 0], [2, 0], [1, 1]])
w2 = np.array([[-1, 0], [0, 1], [-1, 1]])
w3 = np.array([[-1, -1], [0, -1], [0, -2]])

cov = np.zeros((3, 2, 2))
cov[0] = np.cov(w1.T, bias=True)
cov[1] = np.cov(w2.T, bias=True)
cov[2] = np.cov(w3.T, bias=True)

sw = np.zeros((2, 2))
for i in range(3):
    sw = np.add(sw, cov[i])
sw = (1 / 3) * sw
print('S_w = ')
print(sw)

m = np.zeros((3, 2))
m[0] = np.mean(w1, axis=0)
m[1] = np.mean(w2, axis=0)
m[2] = np.mean(w3, axis=0)
m0 = np.mean(m, axis=0)

sb = np.zeros((2, 2))
for i in range(3):
    sb = np.add(sb, np.outer((m[i] - m0), (m[i] - m0)))
sb = (1 / 3) * sb
print('S_b = ')
print(sb)
```



S_w = 
[[ 0.22222222 -0.03703704]
 [-0.03703704  0.22222222]]

S_b = 
[[0.7654321  0.16049383]
 [0.16049383 0.7654321 ]]



## Q2

### 题目

设有如下两类样本集，其出现概率相等：
$$
\omega_1:\quad\{(0\ 0\ 0)^T,(1\ 0\ 0)^T,(1\ 0\ 1)^T,(1\ 1\ 0)^T\}
\\
\omega_2:\quad\{(0\ 0\ 1)^T,(0\ 1\ 0)^T,(0\ 1\ 1)^T,(1\ 1\ 1)^T\}
$$
用K-L变换，分别把特征空间维数降到二维和一维，并画出样本在该空间中的位置



### 解





