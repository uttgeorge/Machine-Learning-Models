# Exponential Family Distribution

这一部分几乎都是笔记，视频请参考：[白板推导系列(八)-指数族分布](https://www.bilibili.com/video/BV1QW411y7D3?p=5)

## 1. 指数族分布的形式
指数族标准形式：
$$
P(x|\eta) =h(x)\cdot \exp(\underset{线性组合}{\underbrace{\eta^T\phi(x)}}-A(\eta))
$$
* $x\in \mathbb{R}^p$
* $\eta:$ 规范化参数向量, P维，是一个关于参数$\theta$的函数
* $A(\eta):$ log partition function (对数配分函数)
* $\phi(x):$ 充分统计量，只和x有关的函数
* $h(x):$ 只和x有关的函数


1. 什么是配分函数 partition function：
$$
P(x|\theta)=\frac{1}{Z}\hat{P}(x|\theta)
$$
Z: 暂时理解为归一化因子，也就是配分函数。

2. 为什么配分函数叫做 log partition function， 为什么是$A(\eta)$这种形式？
$A(\eta)$可以直接提取出来，于是指数族标准形式可以改写为：
$$\begin{align*}
P(x|\eta) &=h(x)\cdot \exp(\eta^T\phi(x))\cdot \exp(-A(\eta))\\\\
&=\frac{1}{\exp(A(\eta))}h(x)\cdot \exp(\eta^T\phi(x))\\\\
&=\frac{1}{Z}\hat{P}(x|\theta)
\end{align*}$$
所以可以认为$Z=\exp(A(\eta))$， 那么$A(\eta)=log(Z)$, 因为$Z$被称为partition function，所以$A(\eta)$就叫log partition function。

3. 常见指数族分布

    高斯分布， 伯努利分布----> 类别分布， 二项分布----> 多项式分布， 泊松分布， beta， dirichlet， gamma

## 2. 指数族分布的性质


指数族分布具有三大特性：
1. 充分统计量

    $\phi(x)$ 是充分统计量。统计量是指关于样本的函数（例如样本均值，方差等），是对数据的加工。充分统计量是指包含数据所有信息的统计量，能完整的表达总体的特征。

    例：

    一个组数据满足高斯分布，$x_1,x_2,x_3,...,x_n$。我们只需要得到以下两个值，就能描述整个样本（可求得均值，方差）：
$$
\phi(x)=\begin{pmatrix}
\sum_{i=1}^{N}x_i
\\\\
\sum_{i=1}^{N}x_i^2
\end{pmatrix}
$$
对于指数族分布，$\phi(x)$本身具备了充分统计量的性质。

    用处：online learning

2. 共轭

    贝叶斯理论：
$$
P(Z|X) = \frac{P(X|Z)P(Z)}{\int_{Z}P(X|Z)P(Z)dZ}
$$
由于积分无法求出，或者后验的形式太复杂，无法求出期望。

    我们可以采用了近似推断求后验，例如：变分推断，MCMC。

    共轭是求后验的另一种方法，如果一个likelihood具有一个与其共轭的先验概率，那么先验概率和后验概率具有相同的分布。例如：似然函数是二项式分布，那么beta与其共轭，后验概率也是beta分布，只不过两个beta分布参数不同。由此，我们能避免求积分。

3. 最大熵（无信息先验）

    在给定一个限制条件的情况下，未知部分我们假设是等可能发生的。但等可能是无法定量分析，所以引入熵，使其熵值最大。最大熵的本质就是让数据更加随机。

    对于贝叶斯原理，有两种给出先验$P(Z)$的方式：
    * 共轭： 为了计算方便
    * 最大熵： 无信息先验，按照最大熵的原理来提供先验。
    * Jerrif

## 3. 指数族分布的模型

1. 广义线性模型 GLM
    1. 线性组合：$w^Tx$
    2. Link Function: 链接函数，是激活函数的反函数
    3. 指数族分布：$(y|x) \sim 指数族分布$
        例如:
        * 线性回归：$y\sim N(\mu,\Sigma)$
        * 二分类：$y\sim Bernoulli$
        * 柏松回归： $y\sim Possion$
        

2. 概率图模型

    无向图：RBM

3. 变分推断

## 4. 高斯分布的指数族形式

一维高斯分布
$$
P(X|\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big),\ \theta=(\mu,\sigma^2)
$$
注：$\eta$其实也是一个函数，$\eta=\eta(\theta),A(\eta)=A(\eta(\theta))$
所以我们要把上式改写成指数族分布的形式，就是把$\theta$映射到$\eta$上，也就是用$\mu,\sigma^2$来表示$\eta$。

$$\begin{align*}
P(X|\theta)&=\frac{1}{\sqrt{2\pi}\sigma}\exp\bigg\lbrace-\frac{(x-\mu)^2}{2\sigma^2}\bigg\rbrace,\theta=(\mu,\sigma^2)\\\\
&=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\bigg\lbrace-\frac{1}{2\sigma^2}(x^2-2\mu x+\mu^2)\bigg\rbrace\\\\
&=exp\bigg\lbrace\log(2\pi\sigma^2)^{-\frac{1}{2}}\bigg\rbrace\exp\bigg\lbrace-\frac{1}{2\sigma^2}(x^2-2\mu x)-\frac{\mu^2}{2\sigma^2}\bigg\rbrace\\\\
&=exp\bigg\lbrace\log(2\pi\sigma^2)^{-\frac{1}{2}}\bigg\rbrace\exp\bigg\lbrace-\frac{1}{2\sigma^2}\big[\begin{pmatrix}
-2\mu &1
\end{pmatrix}\cdot \begin{pmatrix}
x
\\\\ 
x^2
\end{pmatrix}\big]-\frac{\mu^2}{2\sigma^2}\bigg\rbrace\\\\
&=\exp\bigg\lbrace\big[\underset{\eta^T}{\underbrace{\begin{pmatrix}
\frac{\mu}{\sigma^2} &-\frac{1}{2\sigma^2}
\end{pmatrix}}}\cdot \underset{\phi(x)}{\underbrace{\begin{pmatrix}
x
\\\\ 
x^2
\end{pmatrix}}}\big]-\underset{A(\eta)}{\underbrace{\big[\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log(2\pi\sigma^2)\big]}}\bigg\rbrace\\\\
&=\exp(\eta^T\phi(x)-A(\eta))
\end{align*}
$$

其中，

$$\begin{align*}
&1.\ \eta=\begin{pmatrix}
\eta_1
\\\\ 
\eta_2
\end{pmatrix} =\begin{pmatrix}
\frac{\mu}{\sigma^2}
\\\\ 
-\frac{1}{2\sigma^2}
\end{pmatrix}\ \Rightarrow\ \left\lbrace\begin{matrix}
\sigma^2=-\frac{1}{2\eta_2}
\\\\ 
\mu =-\frac{\eta_1}{2\eta_2}
\end{matrix}\right.\\\\
&2.\ A(\eta)=-\frac{\eta_1^2}{4\eta_2}+\frac{1}{2}\log \big(-\frac{\pi}{\eta_2}\big)\\\\
&3.\ \phi(x)=\begin{pmatrix}
x
\\\\ 
x^2
\end{pmatrix}
\end{align*}
$$

## 5. 对数配分函数 log partition function $A(\eta)$

概率密度函数的积分为1:

$$\begin{align*}
P(x|\eta) &=h(x)\cdot \exp(\eta^T\phi(x))\cdot \exp(-A(\eta))\\\\
&=\frac{1}{\exp(A(\eta))}h(x)\cdot \exp(\eta^T\phi(x))\\\\
&=\frac{1}{Z}\hat{P}(x|\theta)\\\\
\int P(x|\eta)dx &= 1=\frac{1}{\exp(A(\eta))}\int h(x)\cdot \exp(\eta^T\phi(x))dx\\\\
\exp(A(\eta))&=Z=\int h(x)\cdot \exp(\eta^T\phi(x))dx
\end{align*}$$

同时对等式两边的$\eta$求导：

$$\begin{align*}
\exp(A(\eta))A'(\eta)&=\frac{\partial}{\partial \eta} \int h(x)\cdot \exp(\eta^T\phi(x))dx\\\\
\exp(A(\eta))A'(\eta)&=\int h(x)\cdot \exp(\eta^T\phi(x))\phi(x)dx\\\\
A'(\eta)&=\frac{\int h(x)\cdot \exp(\eta^T\phi(x))\phi(x)dx}{\exp(A(\eta))}\\\\
&=\int h(x)\cdot \exp(\eta^T\phi(x)-A(\eta))\phi(x)dx\\\\
&=\int P(x|\eta)\phi(x)dx\\\\
&=E_{P(x|\eta)}[\phi(x)]\\\\
\end{align*}$$

于是得出，对数配分函数的一阶导数$A'(\eta)$等于充分统计量的期望。

而其二阶导数$A''(\eta)=Var[\phi(x)]$，是充分统计量的方差。 证明略。

由由于方差一定为正，所以可知$A(\eta)$一定是凸函数。

* 用之前高斯分布举例：

    对$\phi(x)$求期望：
    $$
    E[\phi(x)]=\begin{pmatrix}
    E(x)
    \\\\ 
    E(x^2)
    \end{pmatrix} 
    $$
    
    已知$E(x)=\mu$，那下面证明$A(\eta)$关于$\eta_1$的一阶导数是$\mu$。
    
    $$\begin{align*}
    A'(\eta)&=\frac{\partial A(\eta)}{\partial \eta_1}\\\\
    &=-\frac{\eta_1}{2\eta_2}\\\\
    &=\mu
    \end{align*}
    $$
    
    
## 6. 极大似然估计

现在从极大似然估计的角度，来理解指数族分布。

设一组数据：$D={x_1,x_2,...,x_N}$，其参数的极大似然估计为：

$$\begin{align*}
\eta_{MLE}&=argmax \log P(X|\eta)\\\\
&=argmax \log \prod_{i=1}^{N} P(x_i|\eta)\\\\
&=argmax \sum_{i=1}^{N} \log  P(x_i|\eta)\\\\
&=argmax \sum_{i=1}^{N} \log  \bigg[h(x_i)\cdot \exp(\eta^T\phi(x_i)-A(\eta))\bigg]\\\\
&=argmax \sum_{i=1}^{N}  \bigg[\log h(x_i)+\eta^T\phi(x_i)-A(\eta)\bigg]\\\\
&因为\log h(x)\ 与\eta 无关，所以可以忽略。\\\\
&=argmax \sum_{i=1}^{N}  \bigg[\eta^T\phi(x_i)-A(\eta)\bigg]\\\\
\\
\frac{\partial }{\partial \eta}\eta_{MLE}&=  \sum_{i=1}^{N}  \bigg[\phi(x_i)-A'(\eta)\bigg]\\\\
&=\sum_{i=1}^{N}  \bigg[\phi(x_i)\bigg]-NA'(\eta)=0\\\\
\\
\Rightarrow A'(\eta_{MLE})&=\frac{1}{N}\sum_{i=1}^{N}  \bigg[\phi(x_i)\bigg]
\end{align*}
$$

我们已知$A(\eta)$和$A'(\eta)$，现求$\eta$， 只需要求$A'(\eta)$的反函数：

$$
\eta_{MLE}={A'}^{-1}(\eta_{MLE})
$$

也就是说，要求$\eta_{MLE}$，我们只需要将**充分统计量**求和，然后再求其反函数即可，无需使用所有数据。这也就是充分统计量的优势。

## 7. 最大熵原理

**最大熵原理是指，在已知一定约束条件下，寻找使其熵值最大的分布。**

1. 信息量：

    一个随机变量的概率为$P$，其信息量为$\log\frac{1}{p}=-\log{p}$。

2. 信息熵

    信息熵其实是信息量的期望：
$$\begin{align*}
E[-\log{p(x)}]&=\int -p(x)\log{p(x)}dx\\\\
&=-\sum p(x)\log{p(x)}\\\\
H[P] &=-\sum p(x)\log{p(x)}
\end{align*}
$$

1. **在无任何信息先验的情况下，最大熵与等可能是等价的**，证明如下：

    假设k个变量x，每个变量的所对应的概率为$p_k$，使总信息熵最大：
$$Max:\ \sum_{i=1}^{k}-p_i\log{p_i}\\\\
s.t.\ \sum_{i=1}^{k}p_i=1$$
这是一个有约束的最优化问题，利用Lagrange Multiplier:
$$
Max:\ L(p,\lambda) = \sum_{i=1}^{k}-p_i\log{p_i}+\lambda (\sum_{i=1}^{k}p_i-1),\ s.t. \lambda>0
$$
等价于：
$$
Min:\ L(p,\lambda) = \sum_{i=1}^{k}p_i\log{p_i}-\lambda (\sum_{i=1}^{k}p_i-1)
$$
求$L(p,\lambda)$关于$p_i$的偏导数，（求$p_1$偏导时，其他p相当于常数）
$$\begin{align*}
\frac{\partial L}{\partial p_i}&=\log{p_i}+1-\lambda=0\\\\
\hat p_i&=\exp(\lambda-1)
\end{align*}
$$
由于$\lambda$是一个常量，所以$p_1=p_2=...=p_k=\frac{1}{k}$。

    所以$P(x)$是均匀分布。

4. **经验分布下的最大熵**

    已知一组数据，$X\in\lbrace x_1,x_2,...,x_N\rbrace$
    > 4.1 经验分布
    > 
    > 是对已知样本的描述，其概率密度函数为：
    > $$
    > \hat P(X=x)=\hat p(x)=\frac{count(x)}{N}
    > $$
    > 该函数的含义是：等于$x$的变量出现的频率。可以求出该分布的期望$E_{\hat P}[X]$，方差$Var_{\hat P}[X]$等。
    > 
    > 令$f(x)$是任意关于x的**函数向量**：
    > $$E_{\hat P}[f(x)]=\Delta$$
    > $\Delta$是一个向量，并且已知，这就是求最大熵的约束条件。
    
    > 4.2 最大熵
    >
    > 信息熵公式：
    > $$\begin{align*}
    > min:\ H[P] &=\sum p(x)\log{p(x)}\\\\
    > s.t.\ &\sum p(x)=1\\\\
    > &E_{ P}[f(x)]=E_{\hat P}[f(x)]=\Delta
    > \end{align*}$$
    > 引入Lagrange Multiplier:
    > $$
    > L(p,\lambda)=\sum p(x)\log{p(x)} + \lambda_0(1-\sum p(x)) + \lambda^T(\Delta-E_{P}[f(x)])
    > $$
    > 对其求导：
    > $$\begin{align*}
    > \frac{\partial}{\partial p(x)}L(p,\lambda)=\sum (\log{p(x)} + 1)- \sum \lambda_0 - \sum\lambda^Tf(x)&=0\\\\
    > \sum \Big(\log{p(x)} + 1-  \lambda_0 - \lambda^Tf(x)\Big)&=0
    > \end{align*}
    > $$
    > 由于$p(x)$和$\lambda^T$实际上是向量，所以要保证每一项都为0，则：
    > $$\log{p(x)} + 1-  \lambda_0 - \lambda^Tf(x)=0$$
    > 得到：
    > $$\begin{align*}
    > \log{p(x)} &= \lambda^Tf(x)+\lambda_0-1\\\\
    > p(x)&=\exp \lbrace \lambda^Tf(x)-（1-\lambda_0） \rbrace\\\\
    > &=\exp\lbrace\eta^T\phi(x)-A(\eta)\rbrace
    > \end{align*}
    > $$
    > 由此可知，在有约束条件（经验分布）情况下，使其熵值最大的概率分布为指数族分布。

## 8. 共轭分布
待补充