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
 2. 利用所有的features $x_i$ 以及 $r_{i,t}$，拟合一棵CART回归树，得到$J$个叶子节点。**$J$的个数通常介于[8,32]。**
 3. 每一个叶子节点定义一个输出值$c$，使损失函数最小：
    $$c_{t,j}=\underset{c}{argmin} \sum_{x_i\in R_{t,j}} L(y_i,F_{t-1}({x_i})+c)$$
    当损失函数是Square Loss时，$\sum_{x_i\in R_{t,j}} y_i-F_{t-1}({x_i})-c = 0$，其中$y_i-F_{t-1}({x_i})$是Residual，$c_{t,j}$就等于一个叶子节点中所有Residual的均值。（此仅为特殊情况。）
 4. 利用叶子节点输出值来更新模型：
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
2. Log Likelihood: 因为GBDT要求Gradient，就必须使用连续变量。而分类问题是离散的，所以通过$log(odds)$将二者结合起来。
    
    类似于logistic regression，可以使用最大似然估计作为目标函数：
    $$
    L(y,F)=-yF + \log \big(1+e^{F}\big)
    $$
推导过程如下：

$$\begin{align*}
令 Z_i&=\log(odds_i)\\\\
&= \underset{Z_i}{argmin} \sum_{i=1}^{N} \bigg\lbrace-y_iZ_i + \log \big[1+e^{Z_i}\big]\bigg\rbrace\\\\
&  \sum_{i=1}^{N} \frac{\partial}{\partial Z_i}\bigg\lbrace-y_iZ_i + \log \big[1+e^{Z_i}\big]\bigg\rbrace\\\\
=& \sum_{i=1}^{N} \bigg\lbrace-y_i + \frac{e^{Z_i}}{1+e^{Z_i}}\bigg\rbrace\\\\
=& \sum_{i=1}^{N} \bigg\lbrace-y_i + P_i\bigg\rbrace\\\\
\end{align*}
$$

## 3. 算法步骤


### 3.1 Regression

输入数据： $D = \lbrace(x_i,y_i)\rbrace^{n}_{i=1}$

Step 1: Initialize model with a constant value:
    $$
    F_0(x)=\underset{C}{argmin} \sum^{n}_{i=1}L(y_i,C)
    \\\\
    \rightarrow \frac{\sum y_i}{n}=C=\bar{y}=  F_0(x)
    $$
    当损失函数是Square Loss时，$F_0(x)$等于实际值的均值。
    
Step 2: 

for t = 1 to T: （其中t指第t棵树）
>>A. 计算负梯度：
>>$$
>>r_{i,t}=-\bigg[\frac{\partial{L(y_i,F_{t-1}(x_i))}}{\partial{F_{t-1}(x_i)}}\bigg],\ for\ i=1,2,3,...,n
>>$$ 
>>
>>B. Fit a regression tree to the $r_{i,t}$ value.
>>
>>C. 每一个叶子节点定义一个输出值$c$，使损失函数最小：
>>$$c_{t,j}=\underset{c}{argmin} \sum_{x_i\in R_{t,j}} L(y_i,F_{t-1}({x_i})+c)$$
>>当损失函数是Square Loss时，$\sum_{x_i\in R_{t,j}} y_i-F_{t-1}({x_i})-c = 0$，其中$y_i-F_{t-1}({x_i})$是Residual，$c_{t,j}$就等于一个叶子节点中所有Residual的均值。（此仅为特殊情况。）
>>
>>D. 利用叶子节点输出值来更新模型：
    $$F_{t}(x)=F_{t-1}(x)+v\sum_{j=1}^{J}c_{t,j}*I(x\in R_{t,j})$$
    其中，$v$是学习率。
    
Step 3: 输出$F_T(x)=F_0(x)+v\sum_{t=1}^{T}\sum_{j=1}^{J}c_{t,j}*I(x\in R_{t,j})$

### 3.2 Classification

输入数据： $D = \lbrace(x_i,y_i)\rbrace^{n}_{i=1}$

Step 1: Initialize model with a constant value:
    $$
    F_0(x)=\underset{C}{argmin} \sum^{n}_{i=1}L(y_i,C)\\\\
    s.t.\ C=\log(odds),\ y\in\lbrace0,1\rbrace
    $$
  
    
Step 2: 

for t = 1 to T: （其中m指第m棵树）
>>A. 计算负梯度：
>>$$
>>r_{i,t}=-\bigg[\frac{\partial{L(y_i,F_{t-1}(x_i))}}{\partial{F_{t-1}(x_i)}}\bigg]= -y_i + \frac{e^{F_{t-1}(x_i)}}{1+e^{F_{t-1}(x_i)}}
=-y_i + P_i\,\\\\ for\ i=1,2,3,...,n
>>$$ 
>>
>>B. Fit a regression tree to the $r_{i,t}$ value.
>>
>>C. 每一个叶子节点定义一个输出值$c$，使损失函数最小：
>>$$c_{t,j}=\underset{c}{argmin} \sum_{x_i\in R_{t,j}} L(y_i,F_{t-1}({x_i})+c)$$
>>由于计算量过大，通过2nd-order Taylor Series将公式展开，再通过求导得到最优解。
>>$$\begin{align*}
>>c_{t,j}&=\frac{\sum{-\frac{d}{dF()}L(y_i,F_{t-1}(x_i))}}{\sum{\frac{d^2}{dF()^2}L(y_i,F_{t-1}(x_i))}}\\\\
>>&=\frac{\sum -y_i+\frac{e^{F_{t-1}(x_i)}}{1+e^{F_{t-1}(x_i)}}}{\sum{\frac{e^{F_{t-1}(x_i)}}{(1+e^{F_{t-1}(x_i)})^2}}} & ①\\\\
>>&=\frac{\sum -y_i-P_{i,t-1}}{\sum{P_{i,t-1}(1-P_{i,t-1})}} &②\\\\
>>\end{align*}
>>$$
>>
>>D. 利用叶子节点输出值来更新模型：
    $$F_{t}(x)=F_{t-1}(x)+v\sum_{j=1}^{J}c_{t,j}*I(x\in R_{t,j})$$
    其中，$v$是学习率。

具体推导：略
    
Step 3: 输出$F_t(x)$

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
## 4.正则化 Regularization
1. Learning Rate: $v$
2. Subsample: 不放回的取sample中的一部分数据进行训练
3. Pruning: 对CART决策树剪枝

## 5. 总结

### 5.1 GBDT的核心
Gradient。我们可以用残差来学习下一棵树，除第一棵树，其余树全部由Residual决定。但Residual的问题在于cost function是Square Loss Function，很难处理回归以外的问题。而用Gradient的概念去近似，只要cost function可导就行。

### 5.2 GBDT中的Decision Tree是什么？

回归树，因为GBDT会累加所有树的结果，分类树无法完成此任务。

### 5.3 Gradient Descent 和 Gradient Boosting 的区别

Gradient Descent（更新参数）： 对于一个单独的损失函数，找到使损失函数最小的参数值。

Gradient Boosting（更新函数）： 在函数空间中，找到使梯度下降最快的函数，并将此函数线性添加到之前得到的最有函数中(线性组合)。

### 5.4 GBDT 和 Adaboost 的区别

* 损失函数不同： Adaboost 通过提升错分数据的权重来衡量不足，不断加入弱分类器进行boosting。 GBDT 则是通过residual来衡量模型的不足，并通过不断加入新的树在负梯度方向上最快的减少residual。
* weak learner 不同： Adaboost 的弱学习器是只有一层的decision tree，称作decision stump。 GBDT的弱学习器是regression tree。

### 5.5 GBDT 和 Random Forest 的区别
相同点：
1. 多棵树组成
2. 最终的结果由多棵树一起决定

不同点：
1. RF的子树可以是分类树，也可以是回归树，而GBDT只能是回归树
2. RF是bagging，GBDT是boosting
3. RF可以并行(Parallel)，GBDT只能串行(Serial)
4. RF对异常值不敏感，GBDT更敏感，可以改用Huber loss
5. 相对于RF，GBDT更容易overfitting
6. GBDT准确率更高，相对于SVM等

### 5.6 为什么在分类问题中，使用log(odds)？

1. $p$的取值范围只能是$0-1$，$odds$取值范围是$[o,+\infty)$，而$\log(odds)$取值范围是$(-\infty,+\infty)$
2. 用$\log(odds)$可以构建线性模型，而GBDT正好是一个线性相加的模型。
