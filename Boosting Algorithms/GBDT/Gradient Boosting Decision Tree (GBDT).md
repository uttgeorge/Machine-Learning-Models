# Gradient Boosting Decision Tree (GBDT)

## 1. 简介
GBDT是利用boosting策略，通过每一个weak learner对残差的拟合最终实现高度精准的预测。具体来讲，当训练一个weak learner时，预测值与实际值之间有一定的差值，这个差值被称作Residual或者Pseudo Residual，后文会细讲。由于GBDT是一个加法模型，在不改变已经训练好的weak learner的情况下，进一步训练第二个weak learner，用该学习器来尽可能拟合前文提到的Residual。Iterate这个过程，把每个weak learner相加得到最终的学习器。

## 2. 数学推导

### 2.1 加法模型

GBDT是一个加法模型：

$$
\hat{y_i}=\sum_{k=1}^{K}f_k(x_i)=F_K(x_i)
$$
$f_k$属于Hilbert函数空间

$$
\hat{y_i}^t=\hat{y_i}^{t-1}+f_t(x_i)
$$

### 2.2 Residual vs. Gradient

Residual是前一个模型没有完全拟合的部分，GBDT的核心其实是用Gradient来代替residual，所以也被称为Pseudo Residual。他们之间的联系是通过 Taylor Series 实现的。

首先定义一个损失函数：
$$\begin{align*}
L^{(t)} &= \sum_{i=1}^{N}l(y_i,\hat{y_i}^t)\\\\
&=\sum_{i=1}^{N}l(y_i,\hat{y_i}^{t-1}+f_t(x_i))
\end{align*}$$

对该函数进行一阶泰勒展开，其中$t-1$时刻前（包括t-1）的模型是已知的，可以视作常数，不影响函数优化：
$$\begin{align*}
L^{(t)} & \approx \sum_{i=1}^{N} \bigg[ l(y_i,\hat{y_i}^{t-1})+l'(y_i,\hat{y_i}^{t-1}) * f_t(x_i)\bigg] \\\\
& \approx \sum_{i=1}^{N} \bigg[ l'(y_i,\hat{y_i}^{t-1}) * f_t(x_i)\bigg]\\\\
& \approx \sum_{i=1}^{N} \bigg[ g_i * f_t(x_i)\bigg] 
\end{align*}
$$
其中$g_i= l'(y_i,\hat{y_i}^{t-1})$。当损失函数是least square时, $g_i= 2(\hat{y_i}^{t-1}-y_i)$,刚好是Residual。

### 2.3 Decision Tree

 Gradient Boost 是一类算法，但通常使用决策树作为Weak Leaner。注意，**必须使用CART回归树，即便是分类问题**。
 当第一轮的CART完成预测后，计算负梯度，利用第二棵树来fit负梯度。
 具体如下：
 
 1. 计算负梯度：
    $$r_{i,t}=-g_{i,t}=-\bigg[ \frac{\partial{l(y_i,\hat{y_i}^{t-1})}}{\partial{\hat{y_i}^{t-1}}}\bigg]=-\bigg[ \frac{\partial{l(y_i,F_{t-1}(x_i))}}{\partial{F_{t-1}(x_i)}}\bigg]$$
    其中，$i$表示第i个样本，$t$表示第t棵树。
 3. 利用所有的features $x_i$ 以及 $r_{i,t}$，拟合一棵CART回归树，得到$J$个叶子节点。**$J$的个数通常介于[8,32]。**
 4. 每一个叶子节点使损失函数最小的输出：
    $$c_{t,j}=\underset{c}{argmin} \sum_{x_i\in R_{t,j}} L(y_i,F_{t-1}({x_i})+c)$$
 4. 利用叶子节点输出来更新模型：
    $$F_{t}(x)=F_{t-1}(x)+v\sum_{j=1}^{J}c_{t,j}*I(x\in R_{t,j})$$
    其中，$v$是学习率。

### 2.4 损失函数

#### 2.4.1 回归问题

常用的损失函数有：
1. Square Loss: $L(y,F) = \frac{1}{2}(y-F)^2$
2. Absolute Loss: $L(y,F) =|y-F|$
3. Huber Loss: $ L(y,F) = \left\lbrace\begin{matrix}\begin{align*}
&\frac{1}{2}(y-F)^2 &|y-F| \le \delta  \\\\ &\delta(|y-F|-\delta / 2) &|y-F|>\delta \end{align*}\end{matrix}\right. $

Square Loss求一阶导后，正好是Residual，但是该损失函数对于Outliers过于敏感，通常使用另外两种损失函数。

#### 2.4.2 分类问题

常用损失函数有：
1. Exponential Loss: 变为Adaboost
2. Log Likelihood: 


## 3. 算法步骤









<!--由Boosting策略可得到一个加法模型：$F_t(x)=F_{t-1}(x)+f_t(x)$，每次新的模型结果加到之前模型的累加和上。本质上，任何模型都希望与实际值尽可能接近。当建立一个模型后，与实际值之间的差值，可以通过新的模型去拟合，从而使差值越来越小。所以由加法模型可以看出，我们可以把$f_t(x)$看作是对这种差值的一种逼近。

首先定义residual：$t$时刻的residual等于实际值与$t-1$时刻加法模型的累计值之间的差值，$R_t=y-F_{t-1}(x)$。也就是说$R_t$表示$t-1$次迭代后，系统中的residual。那么当第$t$次迭代时，就要用第$t$棵决策树去尽可能的拟合residual: $R_t$。当$f_t(x)$的值要与$R_t$的值越接近，$L(y,F_{t}(x))$越小。

由此可知，$t$时刻最优的 weak learner $f^{\star}_{t}$ 应该是（为了展示方便，暂时忽略正则项）：
$$\begin{align*}
f^{\star}_{t}&=\underset{f_t}{argmin}\ L(y_i,F_{t}(x_i))\\\\
&=\underset{f_t}{argmin}\ L(y_i,F_{t-1}(x)+f_t(x_i))
\end{align*}
$$
当损失函数是least square时，
$$
L(y,F_{t-1}(x)+f_t(x)) = \frac{1}{2}(R_t-f_t(x))^2
$$
令$f_t(x)$为residual即可。 但如果是其他损失函数，则计算困难。

由此，引入gradient descent来使$f_t(x)$逐渐接近$R_t$。-->

### 1.1 Gradient Descent 和 Gradient Boosting 的区别

Gradient Descent（更新参数）： 对于一个单独的损失函数，找到使损失函数最小的参数值。

Gradient Boosting（更新函数）： 在函数空间中，找到使梯度下降最快的函数，并将此函数线性添加到之前得到的最有函数中(线性组合)。

### 1.2 GBDT 和 Adaboost 的区别

* 损失函数不同： Adaboost 通过提升错分数据的权重来衡量不足，不断加入弱分类器进行boosting。 GBDT 则是通过residual来衡量模型的不足，并通过不断加入新的树在负梯度方向上最快的减少residual。
* weak learner 不同： Adaboost 的弱学习器是只有一层的decision tree，称作decision stump。 GBDT的弱学习器是regression tree。

## 2. 数学推导

### 2.1 目标函数

由1.已知，

## 3. 算法

### 3.1 回归 Regression

1. 输入数据： $D = \lbrace(x_i,y_i)\rbrace^{n}_{i=1}$
2. Loss Function: $L(y_i,F(x))=\frac{1}{2}(y_i-F(x_i))^2$
3. Step 1: Initialize model with a constant value:
    $$
    F_0(x)=\underset{C}{argmin} \sum^{n}_{i=1}L(y_i,C)
    \\\\
    \rightarrow \frac{\sum y_i}{n}=C=\bar{y}=  F_0(x)
    $$
    $F_0(x)$等于预测值的均值。
4. Step 2: for m = 1 to M: （其中m指第m棵树）
    A. 计算梯度：
    $$
    r_{i,m}=-\bigg[\frac{\partial{L(y_i,F_{m-1}(x_i))}}{\partial{F_{m-1}(x_i)}}\bigg],\ for\ i=1,2,3,...,n
    $$ 