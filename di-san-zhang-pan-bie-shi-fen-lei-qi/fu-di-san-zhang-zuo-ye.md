# 附 第三章作业

## 作业（1）

### 题目

在一个10类的模式识别问题中，有3类单独满足多类情况1，其余的类别满足多类情况2。问该模式识别问题所需判别函数的最少数目是多少？



### 解

- 对于3类满足情况1的，将剩下的7类合看作一类，则实际上是4分类问题，需要4个判别函数
- 剩下7个类别为情况2，需要$$\tfrac{M(M-1)}{2}$$条判别函数
- 故总共需要：

$$
4 + \frac{7\times 6}{2} = 25
$$



{% hint style="warning" %}

这里24个也是可以的，不再做一次额外的判断

{% endhint %}



## 作业（2）

### 题目

一个三类问题，其判别函数如下：

 $$d_1(x)=-x_1, d_2(x)=x_1+x_2-1, d_3(x)=x_1-x_2-1$$

1. 设这些函数是在多类情况1条件下确定的，绘出其判别界面和每一个模式类别的区域。

2. 设为多类情况2，并使：$$d_{12}(x)= d_1(x), d_{13}(x)= d_2(x), d_{23}(x)= d_3(x)$$。绘出其判别界面和多类情况2的区域。

3. 设$$d_1(x), d_2(x)$$和$$d_3(x)$$是在多类情况3的条件下确定的，绘出其判别界面和每类的区域。



### 解

#### Q1

![](D:\program\python\PRMLClass\image\判别函数1.png)

#### Q2

![](D:\program\python\PRMLClass\image\判别函数2.png)

#### Q3

$$
\begin{align}
\because\ &d_1(x)=-x_1
\\
& d_2(x)=x_1+x_2-1
\\
& d_3(x)=x_1-x_2-1
\\
\\
\therefore\ & d_{12}(x) = d_1(x) - d_2(x) = -2x_1-x_2+1=0
\\
& d_{13}(x) = d_1(x)-d_3(x) = -2x_1+x_2+1
\\
& d_{23}(x) = d_2(x)-d_3(x) = 2x_2 = 0
\end{align}
$$

![](D:\program\python\PRMLClass\image\判别函数3.png)



## 作业（3）

### 题目

两类模式，每类包括 5 个 **3 维**不同的模式向量，且良好分布。如果它们是线性可分的，问权向量至少需要几个系数分量？

假如要建立二次的多项式判别函数，又至少需要几个系数分量？（设模式的良好分布不因模式变化而改变）



### 解

代入公式
$$
N_w = C_{n+r}^r=\frac{(n+r)!}{r!n!}
$$
则当线性可分时，r=1，n=3，$$N_w$$=4

当采用二次多项式判别函数时，r=2，n=3，$$N_w$$=10

{% hint style="warning" %}

只看维度与最高幂

{% endhint %}



## 作业（4）

### 题目

用感知器算法求下列模式分类的解向量$$\boldsymbol{w}$$:
$$
\omega_1:\{(0\ 0\ 0)^T,(1\ 0\ 0)^T,(1\ 0\ 1)^T,(1\ 1\ 0)^T\}
\\
\omega_2:\{(0\ 0\ 1)^T,(0\ 1\ 1)^T,(0\ 1\ 0)^T,(1\ 1\ 1)^T\}
$$


### 解

首先讲属于$$\omega_2$$的样本乘以-1，写成增广形式：
$$
\begin{align}
&x_1 = (0\ 0\ 0\ 1)^T
\\
&x_2 = (1\ 0\ 0\ 1)^T
\\
&x_3 = (1\ 0\ 1\ 1)^T
\\
&x_4 = (1\ 1\ 0\ 1)^T
\\
&x_5 = (0\ 0\ -1\ -1)^T
\\
&x_6 = (0\ -1\ -1\ -1)^T
\\
&x_7 = (0\ -1\ 0\ -1)^T
\\
&x_8 = (-1\ -1\ -1\ -1)^T
\end{align}
$$
接下来开始迭代，感知器算法的一般表达：
$$
w(k+1)=
\begin{cases}
w(k) & w^T(k)x^k>0
\\
w(k)+Cx^k &w^T(k)x^k \leq 0
\end{cases}
$$


由于步数较多，此处直接给出程序运行结果：

[[0. 0. 0. 1.]
 [1. 0. 0. 1.]
 [1. 0. 1. 1.]
 [1. 1. 0. 1.]
 [0. 0. 1. 1.]
 [0. 1. 1. 1.]
 [0. 1. 0. 1.]
 [1. 1. 1. 1.]]

init w=[0. 0. 0. 0.]

epoch0:
  for x0=[0. 0. 0. 1.],label=1, now w=[0. 0. 0. 0.], predicted_label=0.0,update w=[0. 0. 0. 1.]
  for x1=[1. 0. 0. 1.],label=1, now w=[0. 0. 0. 1.], predicted_label=1.0,keep w
  for x2=[1. 0. 1. 1.],label=1, now w=[0. 0. 0. 1.], predicted_label=1.0,keep w
  for x3=[1. 1. 0. 1.],label=1, now w=[0. 0. 0. 1.], predicted_label=1.0,keep w
  for x4=[0. 0. 1. 1.],label=-1, now w=[0. 0. 0. 1.], predicted_label=1.0,update w=[ 0.  0. -1.  0.]
  for x5=[0. 1. 1. 1.],label=-1, now w=[ 0.  0. -1.  0.], predicted_label=-1.0,keep w
  for x6=[0. 1. 0. 1.],label=-1, now w=[ 0.  0. -1.  0.], predicted_label=0.0,update w=[ 0. -1. -1. -1.]
  for x7=[1. 1. 1. 1.],label=-1, now w=[ 0. -1. -1. -1.], predicted_label=-1.0,keep w
epoch1:
  for x0=[0. 0. 0. 1.],label=1, now w=[ 0. -1. -1. -1.], predicted_label=-1.0,update w=[ 0. -1. -1.  0.]
  for x1=[1. 0. 0. 1.],label=1, now w=[ 0. -1. -1.  0.], predicted_label=0.0,update w=[ 1. -1. -1.  1.]
  for x2=[1. 0. 1. 1.],label=1, now w=[ 1. -1. -1.  1.], predicted_label=1.0,keep w
  for x3=[1. 1. 0. 1.],label=1, now w=[ 1. -1. -1.  1.], predicted_label=1.0,keep w
  for x4=[0. 0. 1. 1.],label=-1, now w=[ 1. -1. -1.  1.], predicted_label=0.0,update w=[ 1. -1. -2.  0.]
  for x5=[0. 1. 1. 1.],label=-1, now w=[ 1. -1. -2.  0.], predicted_label=-1.0,keep w
  for x6=[0. 1. 0. 1.],label=-1, now w=[ 1. -1. -2.  0.], predicted_label=-1.0,keep w
  for x7=[1. 1. 1. 1.],label=-1, now w=[ 1. -1. -2.  0.], predicted_label=-1.0,keep w
epoch2:
  for x0=[0. 0. 0. 1.],label=1, now w=[ 1. -1. -2.  0.], predicted_label=0.0,update w=[ 1. -1. -2.  1.]
  for x1=[1. 0. 0. 1.],label=1, now w=[ 1. -1. -2.  1.], predicted_label=1.0,keep w
  for x2=[1. 0. 1. 1.],label=1, now w=[ 1. -1. -2.  1.], predicted_label=0.0,update w=[ 2. -1. -1.  2.]
  for x3=[1. 1. 0. 1.],label=1, now w=[ 2. -1. -1.  2.], predicted_label=1.0,keep w
  for x4=[0. 0. 1. 1.],label=-1, now w=[ 2. -1. -1.  2.], predicted_label=1.0,update w=[ 2. -1. -2.  1.]
  for x5=[0. 1. 1. 1.],label=-1, now w=[ 2. -1. -2.  1.], predicted_label=-1.0,keep w
  for x6=[0. 1. 0. 1.],label=-1, now w=[ 2. -1. -2.  1.], predicted_label=0.0,update w=[ 2. -2. -2.  0.]
  for x7=[1. 1. 1. 1.],label=-1, now w=[ 2. -2. -2.  0.], predicted_label=-1.0,keep w
epoch3:
  for x0=[0. 0. 0. 1.],label=1, now w=[ 2. -2. -2.  0.], predicted_label=0.0,update w=[ 2. -2. -2.  1.]
  for x1=[1. 0. 0. 1.],label=1, now w=[ 2. -2. -2.  1.], predicted_label=1.0,keep w
  for x2=[1. 0. 1. 1.],label=1, now w=[ 2. -2. -2.  1.], predicted_label=1.0,keep w
  for x3=[1. 1. 0. 1.],label=1, now w=[ 2. -2. -2.  1.], predicted_label=1.0,keep w
  for x4=[0. 0. 1. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
  for x5=[0. 1. 1. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
  for x6=[0. 1. 0. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
  for x7=[1. 1. 1. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
epoch4:
  for x0=[0. 0. 0. 1.],label=1, now w=[ 2. -2. -2.  1.], predicted_label=1.0,keep w
  for x1=[1. 0. 0. 1.],label=1, now w=[ 2. -2. -2.  1.], predicted_label=1.0,keep w
  for x2=[1. 0. 1. 1.],label=1, now w=[ 2. -2. -2.  1.], predicted_label=1.0,keep w
  for x3=[1. 1. 0. 1.],label=1, now w=[ 2. -2. -2.  1.], predicted_label=1.0,keep w
  for x4=[0. 0. 1. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
  for x5=[0. 1. 1. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
  for x6=[0. 1. 0. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
  for x7=[1. 1. 1. 1.],label=-1, now w=[ 2. -2. -2.  1.], predicted_label=-1.0,keep w
w = [ 2. -2. -2.  1.]

最终得到权向量为$$w=(2,-2,-2,1)^T$$，故判别函数为：
$$
d(x) = 2x_1-2x_2-2x_3+1
$$



上述结果的代码如下：

```python
import numpy as np


def perceptron_train(_patterns, _labels, learning_rate, epochs):
    _patterns = np.hstack((patterns, np.ones((_patterns.shape[0], 1))))
    print(f'{_patterns}\n')
    _w = np.zeros(_patterns.shape[1])
    print(f'init w={_w}\n')

    for epoch in range(epochs):
        print(f'epoch{epoch}:')
        w_update = False
        for i, pattern in enumerate(_patterns):
            predicted_label = np.sign(np.dot(pattern, _w))

            print(f'  for x{i}={pattern},label={_labels[i]}, now w={_w}, predicted_label={predicted_label},', end='')
            if predicted_label != _labels[i]:
                _w += learning_rate * _labels[i] * pattern
                w_update = True
                print(f'update w={_w}')
            else:
                print(f'keep w')

        if not w_update:
            break

    return _w


patterns = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                     [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])

w = perceptron_train(patterns, labels, 1, 10)

print(f'w = {w}'')
```




## 作业（5）

### 题目

用多类感知器算法求下列模式的判别函数：
$$
\omega_1:(-1,-1)^T
\\
\omega_2:(0,0)^T
\\
\omega_3:(1,1)^T
$$

### 解

由于递推字数较多，此处仍然直接给出程序运行结果：

[[-1. -1.  1.]
 [ 0.  0.  1.]
 [ 1.  1.  1.]]

init w=
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]

epoch0:
  for x1=[-1. -1.  1.], now w=[0. 0. 0. 0. 0. 0. 0. 0. 0.], max_label=[0 1 2], 
    update w=[-1. -1.  1.  1.  1. -1.  1.  1. -1.]

  for x2=[0. 0. 1.], now w=[-1. -1.  1.  1.  1. -1.  1.  1. -1.], max_label=[0], 
    update w=[-1. -1.  0.  1.  1.  0.  1.  1. -2.]

  for x3=[1. 1. 1.], now w=[-1. -1.  0.  1.  1.  0.  1.  1. -2.], max_label=[1], 
    update w=[-2. -2. -1.  0.  0. -1.  2.  2. -1.]

epoch1:
  for x1=[-1. -1.  1.], now w=[-2. -2. -1.  0.  0. -1.  2.  2. -1.], max_label=[0], 
    keep w

  for x2=[0. 0. 1.], now w=[-2. -2. -1.  0.  0. -1.  2.  2. -1.], max_label=[0 1 2], 
    update w=[-2. -2. -2.  0.  0.  0.  2.  2. -2.]

  for x3=[1. 1. 1.], now w=[-2. -2. -2.  0.  0.  0.  2.  2. -2.], max_label=[2], 
    keep w

epoch2:
  for x1=[-1. -1.  1.], now w=[-2. -2. -2.  0.  0.  0.  2.  2. -2.], max_label=[0], 
    keep w

  for x2=[0. 0. 1.], now w=[-2. -2. -2.  0.  0.  0.  2.  2. -2.], max_label=[1], 
    keep w

  for x3=[1. 1. 1.], now w=[-2. -2. -2.  0.  0.  0.  2.  2. -2.], max_label=[2], 
    keep w

w = [[-2. -2. -2.]
 [ 0.  0.  0.]
 [ 2.  2. -2.]]



综上，得到的判别函数为：
$$
d_1(x) = -2x_1-2x_2-2
\\
d_2(x) = 0
\\
d_3(x) = 2x_1+2x_2-2
$$
所用程序如下：

```python
import numpy as np


def perceptron_train(_patterns, _labels, learning_rate, epochs):
    _patterns = np.hstack((patterns, np.ones((_patterns.shape[0], 1))))
    print(f'{_patterns}\n')
    _w = np.zeros((_patterns.shape[1], _patterns.shape[0]))
    print(f'init w=\n{_w}\n')

    for epoch in range(epochs):
        print(f'epoch{epoch}:')
        w_update = False
        for i, pattern in enumerate(_patterns):
            _d = np.dot(_w, np.transpose(pattern))
            max_label = np.where(_d == np.max(_d))[0]

            print(f'  for x{i + 1}={pattern}, now w={_w.flatten()}, max_label={max_label}, \n', end='')

            if max_label.__len__() == 1 and max_label[0] == i:
                print(f'    keep w\n')
            else:
                _w += learning_rate * np.outer(labels[i], pattern)
                print(f'    update w={_w.flatten()}\n')
                w_update = True

        if not w_update:
            break

    return _w


patterns = np.array([[-1, -1],
                     [0, 0],
                     [1, 1]])
labels = np.array([[1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, 1]])

w = perceptron_train(patterns, labels, 1, 10)

print(f'w = {w}')
```

