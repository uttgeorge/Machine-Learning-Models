# Exponential Family Distribution

高斯分布

伯努利分布----> 类别分布

二项分布----> 多项式分布

泊松分布

beta

dirichlet

gamma

指数族标准形式：
$$
P(x|\eta) =h(x)\cdot exp(\underset{线性组合}{\underbrace{\eta^T\phi(x)}}-A(\eta))
$$
$x\in \mathbb{R}^p$
$\eta:$ 规范化参数向量,p维
$A(\eta):$ log partition function(配分函数)
$\phi(x):$ 只和x有关
$h(x):$ 只和x有关


配分函数：
$$
P(x|\theta)=\frac{1}{Z}\hat{P}(x|\theta)
$$
Z: 归一化因子，也就是配分函数


指数族分布具有三大特性：
1. 充分统计量

$\phi(x)$ 是充分统计量。统计量是样本的函数（例如样本均值，方差等），是对数据的加工。充分统计量是指包含数据所有信息的统计量，能完整的表达总体的特征。
例：
一个组数据满足高斯分布，$x_1,x_2,x_3,...,x_n$。我们只需要得到以下两个值，就能描述整个样本（求得均值，方差）：
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
我们采用了近似推断，例如：变分推断，MCMC。
共轭是求后验的另一种方法，如果一个likelihood具有一个与其共轭的先验概率，那么先验概率和后验概率具有相同的分布。例如：似然函数是二项式分布，那么beta与其共轭，后验概率也是beta分布，只不过两个beta分布参数不同。所以我们能避免求积分。


3. 最大熵（无信息先验）
在给定一个限制条件的情况下，位置部分我们假设是等可能发生的，但无法定量分析，引入熵，最大熵就是让它更加随机。
给先验的方式：
    * 共轭： 计算方便
    * 最大熵： 无信息先验，按照最大熵的原理来提供先验。


指数族分布的模型：

1. 广义线性模型
    1. 线性组合：$w^Tx$
    2. link function: 激活函数的反函数
    3. 指数族分布：$y|x \sim 指数族分布$
        例如:
        * 线性回归：$y\sim N(\mu,\Sigma)$
        * 二分类：$y\sim Bernoulli$
        * 柏松回归： $y\sim Possion$
        

1. 概率图模型

五向图：RBM

1. 变分推断





## 1. Exponential Distributions

**1. Gauss**
1.1 一维高斯分布
$$
P(X|\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp\Big(\frac{(x-\mu)^2}{2\sigma^2}\Big)
$$
注：$\eta$其实也是一个函数，$\eta=\eta(\theta),A(\eta)=A(\eta(\theta))$
所以我们要把上式改写成指数族分布的形式，就是把$\theta$映射到$\eta$上，也就是用$\mu,\sigma^2$来表示$\eta$。

$$\begin{align*}
P(X|\theta)&=\frac{1}{\sqrt{2\pi}\sigma}exp\Big(\frac{(x-\mu)^2}{2\sigma^2}\Big),\theta=(\mu,\sigma^2)\\\\
&=\frac{1}{\sqrt{2\pi\sigma^2}}exp\Big(\frac{1}{2\sigma^2}(x^2-2\mu x+\mu^2)\Big)\\\\
&=exp\Big(log(2\pi\sigma^2)^{-\frac{1}{2}}\Big)exp\Big(-\frac{1}{2\sigma^2}(x^2-2\mu x-\frac{\mu^2}{2\sigma^2})\Big)\\\\
&=-\frac{1}{2}exp(log2\pi\sigma^2)exp(-\frac{1}{2})
\end{align*}
$$
\frac{\mu^2}{2\sigma^2})


**2. Bernoulli Distribution (0-1 Distribution)**

For a random variable $X$, when it success $x=1$ with probability $p$, fail $x=0$ with probability $q=1-p$.
    $$
    f_X(x)=P(x) = p^x(1-p)^{(1-x)},x=0,1
    $$

* *Expectation:* 
 $$ 
 E[X] = \sum_{i=0}^{1}x_iP(x_i)=0\cdot(1-p) + 1\cdot p=p
 $$
 
* *Variance:*

$$\begin{align*}
Var[X] &= E[X^2]-E[X]^2\\\\
&=\sum_{i=0}^{1}x_i^2P(x_i)-p^2\\\\
&=0^2\cdot (1-p)+1^2\cdot p-p^2\\\\
&=(1-p)p\\\\
&=pq\\\\
\end{align*}
$$


**3. Binomial Distribution**

A distribution of the sum of n times independent Bernoulli trails.
* Every Experience is independent
* Each with the same probability $p$


If the random variable X follows the binomial distribution with parameters $n\in N$ and $p\in [0,1]$, we write $X \sim B(n,p)$. The probability of getting exactly k successes in n independent Bernoulli trials is given by the probability mass function:
$$\begin{align*}
    & P(X=k) = \begin{pmatrix} n \\\\ k\end{pmatrix}p^k(1-p)^{(n-k)}\\\\ 
    where\ &k = 0,1,2,..,n\\\\
    & q=1-p
\end{align*}
$$

* Expectation:

$$\begin{align*}
E[X] &= \sum_{k=0}^{n}k\begin{pmatrix} n \\\\k\end{pmatrix}p^k(1-p)^{(n-k)}\\\\
&= \sum_{k=1}^{n}k\begin{pmatrix} n \\\\k\end{pmatrix}p^k(1-p)^{(n-k)}\\\\
&= \sum_{k=1}^{n}k\frac{n!}{k!(n-k)!}p^k q^{(n-k)}\\\\
&= np \sum_{k=1}^{n} \frac{(n-1)!}{(k-1)!(n-k)!}p^{(k-1)} q^{(n-1)-(k-1)}\\\\
&= np \underset{1}{\underbrace{\sum_{K=0}^{N} \begin{pmatrix} N \\\\K  \end{pmatrix}p^{(K)} q^{(N-K)}}}, N=N-1,K=K-1\\\\
& = np
\end{align*}
$$

* Variance:

$$\begin{align*}
E[X^2]&=\sum_{k=1}^{n}k^2 \frac{n!}{k!(n-k)!}p^kq^{(n-k)}\\\\
&=\sum_{k=1}^{n}k(k-1)\begin{pmatrix} n \\\\k\end{pmatrix} p^kq^{(n-k)}+\underset{np}{\underbrace{\sum_{k=1}^{n}k\begin{pmatrix} n \\\\k  \end{pmatrix}p^kq^{(n-k)}}}\\\\
&=\sum_{k=1}^{n}\frac{n!}{(k-2)!(n-k)!}p^kq^{(n-k)}+np\\\\
&=n(n-1)p^2 \underset{1}{\underbrace{\sum_{k=1}^{n}\frac{(n-2)!}{(k-2)!(n-k)!}p^{(k-2)}q^{(n-k)}}} +np\\\\
&=n^2p^2-np^2+np\\\\
Var[X] &= E[X^2]-E[X]^2\\\\
& =n^2p^2-np^2+np - n^2p^2\\\\
&=np(1-p)\\\\
&=npq
\end{align*}
$$

<!--* Mode and Medium
    * if $np$ is an integer, mean, medium and mode are same-->

* Sum of Binomials

    If $X \sim B(n, p)$ and $Y \sim B(m, p)$ are independent binomial variables with the same probability p, then $X + Y$ is again a binomial variable; its distribution is $Z=X+Y \sim B(n+m, p)$
      

**4. Poisson**

柏松分布描述单位时间内随机事件发生的次数的概率分布。A distribution to expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event. 
The Poisson distribution can also be used for the number of events in other specified intervals such as distance, area or volume.
A discrete random variable X is said to have a Poisson distribution with parameter λ > 0, if, for k = 0, 1, 2, ..., the probability mass function of X is given by:
$$

$$
We write as $X \sim \pi(\lambda)$.



**5. Beta**

**6. Dirichlet**

**7. Gamma**


# tweedie