# SVM

**Mind map:**

If we have the data that are linear separable, we could find a line or hyperplane on high dimension space to separate them into two classes, then find the widest road that separate these data based on the hyperplane we found. The widest road allows us to be more confident to separate new data point.

Because it is extremely hard to solve, we utilize Lagrange Multiplier, Duality and KKT condition to find the best solution. But because of Lagrange Multiplier introduced some unknown variable (we can call it $\alpha$ now), we cannot get the best solution for hyperplane immediately. We can use gradient descent to find the best $\alpha$ iteratively, then find the best $w$ and $b$ for the optimal hyperplane. But, there are some constraint that requires sum of $\alpha_iy_i$ equals to 0, so we have to update paired $\alpha$ simultaneously instead of only updating 1 $\alpha$. With the help of SMO algorithm, we can finally find the best solution.

Besides that, because linear separable data sometimes mix with noises, we introduce a slackness variable to measure the cost of noises. This method is called soft margin.

Further more, linear separable assumption is over optimistic, in most case, data are not linear separable. To solve this problem, and thanks to utilizing duality, we can convert our features $x$ into higher dimension. And data are always linear separable on higher dimension. But, it is extremely costly to map data into higher dimension, we can use different kernel functions to solve this problem.  


## Duality

#### 1. Primal Problem: 

$$\begin{align*}
\min:\ &f(x)\\\\
s.t.\ &g_i(x)\le0,i=1,2,...,k\\\\
&h_j(x)=0,j=1,2,...,l
\end{align*}
$$

Introduce **Generalized Lagrange Multiplier(Find extrema under contraints):**

$$\begin{align*}
L(x,\alpha,\beta)&=f(x)+\sum_{i=1}^{k}\alpha_ig_i(x)+\sum_{j=1}^{l}\beta_jh_j(x)\\\\
s.t.\ &\alpha_i\ge 0
\end{align*}
$$




<font color="red">**Based on Generalized Lagrange Function, the primal problem is equivalent to:**</font>



$$\begin{align*}
\underset{x}{\min}\ \underset{\alpha,\beta }{\max}\ &L(x,\alpha,\beta)\\\ 
s.t.\ &\alpha_i\ge0
\end{align*}
$$

$Proof:$
$$\begin{align*}
 &L(x,\alpha,\beta)=f(x)+\sum_{i=1}^{k}\alpha_ig_i(x)+\sum_{j=1}^{l}\beta_jh_j(x),\alpha_i\ge 0\\\\
&if\ g_i(x)>0,\ then\ \underset{\alpha_i\ge 0}{\max}\ L(x,\alpha,\beta) = + \infty;\\\\
&if\ g_i(x)\le0,\ then\ \underset{\alpha_i\ge 0}{\max}\ L(x,\alpha,\beta) = f(x);\\\\
& \underset{x}{\min} \underset{\alpha,\beta}{\max}L(x,\alpha,\beta) = \underset{x}{\min}(+\infty,f(x))=\underset{x}{\min}f(x)\\\\
\end{align*}
$$

$Q.E.D$

#### 2. Dual Problem:

**Definition of duality problems:**

$$\begin{align*}
\max\ \theta(\alpha,\beta)&=\underset{x\in \mathbb{R}^p}{inf} \big\lbrace L(x,\alpha,\beta) \big\rbrace\\\\
s.t.\ &\alpha_i\ge 0
\end{align*}
$$

This function means: let $\alpha$ and $\beta$ be fixed, iterate every $x$ to find the infimum $L(x,\alpha,\beta)$.

**If** $x^{\star}$ **is the solution of the primal problem, and** $\alpha^{\star}$, $\beta^{\star}$ **are the solutions of the dual problem, then** 

$$f(x^{\star})\ge\theta(\alpha^{\star},\beta^{\star})$$

$Proof:$

$$\begin{align*}
\max\ \theta(\alpha^{\star},\beta^{\star})&=\underset{x\in \mathbb{R}^p}{inf} \big\lbrace L(x,\alpha,\beta) \big\rbrace\\\\
&\le L(x^{\star},\alpha^{\star},\beta^{\star})\\\\
&= f(x^{\star})+\sum_{i=1}^{k}\alpha_i^{\star}g_i(x^{\star})+\sum_{j=1}^{l}\beta_j^{\star}h_j(x^{\star}),\ s.t.\ \alpha^{\star}\ge 0,\ g_i(x)\le 0, h_j(x)=0\\\\
&\le f(x^{\star})
\end{align*}
$$

$Q.E.D$

This is so called: **Weak Duality.**

## KKT

#### 1. Definition 
$$G=f(x^{\star})-\theta(\alpha^{\star},\beta^{\star})\ge 0$$
where G is the <font color="red">**Duality Gap**</font> between primal problem and dual problem.
Under some condition, G=0. And this is called **strong duality.**

#### 2. Strong Duality Theorem
If $f(x)$ is a convex function, and $g(x)=A^Tx+b,\ h(x)=C^Tx+d$,
then the duality gap between primal and dual problem for this optimization problem is 0.
In other word:
$$
f(x^{\star})= \theta(\alpha^{\star},\beta^{\star})
$$

#### 3. KKT Condition (Karush-Kuhn-Tucker)

1. $L$ is derivative, and can find an optimal solution. (Convex)
    $
    \nabla_xL=0,\ \nabla_{\alpha}L=0,\ \nabla_{\beta}L=0
    $
    
2. $\alpha_i\ge 0$

3. $g_i(x) \le 0, h_j(x)=0$
4. $\alpha_i \cdot g_i(x) =0,i=1,2,...,k$


$$
in\ order\ to\ let:\ f(x^{\star})+\sum \alpha_ig_i(x)+0=f(x^{\star})\\\\
because\ \alpha_i \ge 0\ and\ g_i(x) \le 0\\\\
therefore\ \alpha_i g_i(x)=0
$$

<font color="red">**KKT condition if sufficient and necessary for strong duality!**</font>

## Hard Margin

```ruby
SVM deals with Binary Classification problems. 
Suppose our data are Linearly Separable, the discriminant function is:

                                f(x)=sign (w^Tx+b)
```
![](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/SVM/media/1_06GSco3ItM3gwW2scY6Tmg.png)
<font size="2">(Source: https://towardsdatascience.com/svm-feature-selection-and-kernels-840781cc1a6c)</font>

The distance from a point $(x_i,y_i)$ to a hyperplane $w^Tx+b$ is:
$$
d = \frac{||w^Tx_i+b||}{||w||}\\\\
margin = 2d
$$

The basic idea is to find the widest "road" (in 2-D space) between class1 and class2, so that we could be more confident to classify any given new point.

In other words, our goal is to find a a maximum margin so that the distances from points on the margin to the hyperplane are maximized.


**Max Margin Classifier:**

$$\begin{align*}
&\max:margin(w,b)
\\\\
s.t.&\Bigg\lbrace\begin{matrix}
w^Tx_i +b>0,y_i=+1\\\\
w^Tx_i +b<0,y_i=-1
\
\end{matrix}
\\\\
 &\Rightarrow y_i(w^{T}x_{i}+b) > 0
 ,\ \forall i=1,2,...,N
 \end{align*}
$$


#### 1. Suppose there exists a hyperplane, find points as close to the hyperplane as possible, calculate the distances from points to hyperplane:

$$\begin{align*}
margin(w,b)&=\underset{x_i}{\min}distance(w,b,x_i),\ s.t.\ i=1,2,...,N.\ w,b\ are fixed.\\\\
&=\underset{x_i}{\min}\frac{1}{||w||}|w^Tx_i+b|
\end{align*}
$$

#### 2. Find the hyperplane that maximize the margin:

$$\begin{align*}
&\underset{w,b}{\max} \underset{x_i}{\min} \frac{1}{||w||}\left|w^Tx_i+b\right|,\ s.t.\ y_i(w^Tx_i+b)>0\\\\
\Rightarrow &\underset{w,b}{\max} \frac{1}{||w||} \underset{x_i,y_i}{\min}y_i(w^Tx_i+b),\ s.t.\ y_i(w^Tx_i+b)>0\\\\
&Let\ \gamma=\underset{x_i,y_i}{\min} y_i(w^Tx_i+b),\ \exists\  \gamma >0\\\\\
\Rightarrow &\underset{w,b}{\max} \frac{\gamma}{||w||},\ s.t.\ y_i(w^Tx_i+b)\ge \gamma\\\\
&Let\ \gamma = 1\\\\
\Rightarrow &\underset{w,b}{\max} \frac{1}{||w||},\ s.t.\ y_i(w^Tx_i+b)\ge 1
\end{align*}
$$

**Note:**
Why setting $\gamma$ as 1?

Because scaling $w,\ b$ simultaneously does not change the hyperplane $w^T+b$, for the mathematical convenience and for not having multiple solutions of $w,\ b$, we set $\gamma=1$.

#### 3. Convex optimization
$$
\underset{w,b}{\max} \frac{1}{||w||} \Rightarrow \underset{w,b}{\min} ||w|| \Rightarrow \underset{w,b}{\min} \frac{1}{2} ||w||^2,\ s.t.\ y_i(w^Tx_i+b)\ge 1
$$



Now, we are solving a **Convex Optimization Problem.**
$$\begin{align*}
&\underset{w,b}{\min} \frac{1}{2} ||w||^2\\\\
s.t.\ &y_i(w^Tx_i+b)\ge 1,\ for\ i=1,2,...,N.\\\\
& There\ are\ N\ Constraints
\end{align*}
$$

* **Primal Problem:**

$$\begin{align*}
&\underset{w,b}{\min} \frac{1}{2} ||w||^2\\\\
s.t.\ &1-y_i(w^Tx_i+b)\le 0,\ for\ i=1,2,...,N.\\\\
\Rightarrow & \underset{w,b}{\min} \underset{\alpha}{\max}\ L(w,b,\alpha)\\\\
=& \underset{w,b}{\min} \underset{\alpha}{\max}\ \frac{1}{2} ||w||^2+\sum_{i=1}^{N}\alpha_i (1-y_i(w^Tx_i+b)),\ s.t.\ \alpha_i\ge0
\end{align*}
$$

* **Dual Problem:**

$$
\underset{\alpha}{\max} \underset{w,b}{\min}  \frac{1}{2} ||w||^2+\sum_{i=1}^{N}\alpha_i (1-y_i(w^Tx_i+b)),\ s.t.\ \alpha_i\ge0
$$



#### 3.1 Solve $\underset{w,b}{\min}L(w,b,\alpha)$

$$
\underset{w,b}{\min}L(w,b,\alpha) = \underset{w,b}{\min}  \frac{1}{2} ||w||^2+\sum_{i=1}^{N}\alpha_i (1-y_i(w^Tx_i+b))
$$

Calculate the derivative of $L$.

$$\begin{align*}
&\frac{\partial L}{\partial w}=w-\sum_{i=1}^{N}\alpha_ix_iy_i=0 &\Rightarrow\ & w=\sum_{i=1}^{N}\alpha_ix_iy_i\\\\
&\frac{\partial L}{\partial b}=-\sum_{i=1}^{N}\alpha_iy_i=0 &\Rightarrow\ & \sum_{i=1}^{N}\alpha_iy_i=0\\\\
\end{align*}
$$

$$\begin{align*}
&\underset{w,b}{\min}L(w,b,\alpha)\\\\
=&\frac{1}{2}(\sum_{i=1}^{N}\alpha_ix_iy_i)^T(\sum_{j=1}^{N}\alpha_jx_jy_j)+\sum_{i=1}^{N}\alpha_i-\sum_{i=1}^{N}\alpha_iy_i(\sum_{j=1}^{N}\alpha_jx_jy_j)^Tx_i\\\\
=&\sum_{i=1}^{N}\alpha_i-\frac{1}{2}(\sum_{i=1}^{N}\alpha_ix_iy_i)^T(\sum_{j=1}^{N}\alpha_jx_jy_j)
\end{align*}
$$

#### 3.2 Solve $\underset{\alpha}{\max}\left[\underset{w,b}{\min}L(w,b,\alpha)\right]$

$$\begin{align*}
&\underset{\alpha}{\max}\sum_{i=1}^{N}\alpha_i-\frac{1}{2}(\sum_{i=1}^{N}\alpha_ix_iy_i)^T(\sum_{j=1}^{N}\alpha_jx_jy_j),\ \ \ \ \ s.t.\ \alpha_i\ge0;\ \sum_{i=1}^{N}\alpha_iy_i=0\\\\
\Rightarrow\ &\underset{\alpha}{\min}\frac{1}{2}(\sum_{i=1}^{N}\alpha_ix_iy_i)^T(\sum_{j=1}^{N}\alpha_jx_jy_j)-\sum_{i=1}^{N}\alpha_i\\\\
=\ &\underset{\alpha}{\min}\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_jx_i^Tx_j-\sum_{i=1}^{N}\alpha_i
\end{align*}
$$

$x_i^Tx_j$, this dot product allows us to use kernel function to solve higher dimensional, non-linear separable data.
 
 And under KKT condition, primal and dual problem have strong duality, so that $\underset{\alpha}{\min}\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_jx_i^Tx_j-\sum_{i=1}^{N}\alpha_i=\frac{1}{2}||w^{\star}||^2$, where $w^{\star}$ is the optimal solution for $w$.
    
#### 3.3 Find $w^{\star}$ and $b^{\star}$

*  $w^{\star}=\sum_{i=1}^{N}\alpha_ix_iy_i$

See 3.1.

* $b^{\star} = y_j-\sum_{i=1}^{N}\alpha_iy_ix_i^Tx_j$


$Proof:$

$$\begin{align*}
 &Based\ on\ KKT\ condition\ No.\ 4:\\\\
&\alpha_i(1-y_i(w^Tx_i+b))=0\\\\
\end{align*}
$$

$$\begin{align*}
    1,\ &When\ \alpha_i>0:\\\\
   & y_i(w^Tx_i+b)=1\\\\
    \Rightarrow\ & y_i^2(w^Tx_i+b)=y_i,\ y_i\in \lbrace+1,-1\rbrace\\\\
    \Rightarrow\ & w^Tx_i+b=y_i\\\\
    \Rightarrow\ & b = y_i-w^Tx_i,\ now\ we\ change\ the \ denote\ of\ x_i\ to\ x_k, y_i\ to\ y_k\\\\
    \Rightarrow\ & b^{\star}=y_k-w^Tx_k,\ replace\ w\ with\ its\ optimal\ solution\\\\\
    \Rightarrow\ & b^{\star}=y_k-\sum_{i=1}^{N}\alpha_iy_ix_i^Tx_k\\\\
    2,\ &When\ \alpha_i=0:\\\\
    & y_i(w^Tx_i+b)\ge 0
\end{align*}
$$

$O.E.D$    
    
    
    
## Soft Margin:

   * Allow some noises, so that increase the robustness of our system. 
   
   * Introduced slackness variable $\xi$, which represents **loss**. 
   
   * And now the margin function changes to $y_i(w^Tx_i+b)\ge1-\xi_i,\ s.t.\ \xi\ge0$

![](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/SVM/media/svm_slack.png)

<font size="2">(source: https://www.datasciencecentral.com/profiles/blogs/implementing-a-soft-margin-kernelized-support-vector-machine)</font>

$y_i(w^Tx_i+b)\ge1-\xi_i$

Suppose we have two classes X and O, for class X:

   1. When X is correctly classified (on the right side), means X locates on or outside the margin, then $\xi=0$
   2. When X is incorrectly classified and locates on the side of $y_i(w^Tx_i+b)\ge0$,then $0<\xi\le0$ 
   3. When X is incorrectly classified and locates on the side of  $y_i(w^Tx_i+b)\le0$, then $\xi>1$
   
Based on these 3 conditions, we get the new target function:

$$
\min_{w,b,\xi }:\frac{1}{2}|| w ||^{2}+\underset{loss}{\underbrace{C\sum_{i=1}^{N}\xi_i}}\\\\
s.t. \Big\lbrace\begin{matrix}y_i(w^Tx_i + b)\ge 1-\xi_i
\\\\ \xi_i\ge0\\
\end{matrix}\
\\
$$

The loss function here is called **Hinge Loss**, basically it uses distance to measure **loss**: $\xi$ represents the distance from a point to its corresponding margin $w^Tx+b=1$ when it is miss-classified.

   1. If $w^Tx+b\ge1$, $\xi_i=0$, No loss, correct.
   2. If $w^Tx+b<1$, $\xi_i=1-y_i(w^Tx+b)$
   
So now we have:
$$
\xi_i =\max\lbrace 0,1-y_i(w^Tx_i + b) \rbrace
$$
#### 1. Duality 

* **Primal Problem:**

$$\begin{align*}
&\min_{w,b,\xi }:\frac{1}{2}|| w ||^{2}+C\sum_{i=1}^{N}\xi_i,\ \ \ \ s.t.\ \xi_i\ge0,\ y_i(w^Tx_i+b)\ge1-\xi_i \\\\
\Rightarrow\ & \underset{w,b,\xi}{\min} \underset{\alpha,\beta}{\max} L(w,b,\xi,\alpha,\beta)\\\\
&=\frac{1}{2}||w||^{2}+C\sum_{i=1}^{N}\xi_i-\sum_{i=1}^{N}\alpha_i\xi_i-\sum_{i=1}^{N}\beta_i\big[y_i(w^Tx_i+b)+\xi_i-1\big]\\\\
&s.t.\ \alpha\ge0,\ \beta \ge0,\ i=1,2,..,N
\end{align*}
$$


* **Dual Problem:**

$$
\underset{\alpha,\beta}{\max} \underset{w,b,\xi}{\min}L(w,b,\xi,\alpha,\beta)
$$

#### 2. KKT Condition Check

#### **2.1. Gradient equals 0**

$$
\underset{w,b,\xi}{\min} L(w,b,\xi,\alpha,\beta)=\frac{1}{2}|| w ||^{2}+C\sum_{i=1}^{N}\xi_i-\sum_{i=1}^{N}\alpha_i\xi_i-\sum_{i=1}^{N}\beta_i\big[y_i(w^Tx_i+b)+\xi_i-1\big]
$$

$$\begin{align*}
&\frac{\partial L}{\partial w}=w-\sum_{i=1}^{N}\beta_ix_iy_i=0 &\Rightarrow\ & w^{\star}=\sum_{i=1}^{N}\beta_ix_iy_i\\\\
&\frac{\partial L}{\partial b}=-\sum_{i=1}^{N}\beta_iy_i=0 &\Rightarrow\ & \sum_{i=1}^{N}\beta_iy_i=0\\\\
&\frac{\partial L}{\partial \xi_i}=C-\alpha_i -\beta_i =0 &\Rightarrow\ & \alpha_i=C-\beta_i\\\\
\\\\
\end{align*}
$$

$$\begin{align*}
\Rightarrow\ &\min L(w^{\star},b^{\star},\xi^{\star},\alpha,\beta)\\\\
=& \frac{1}{2}(\sum_{i=1}^{N}x_iy_i\beta_i)^T(\sum_{j=1}^{N}x_jy_j\beta_j)+C\sum_{i=1}^{N}\xi_i-\sum_{i=1}^{N}(C-\beta_i)\xi_i\\\\&-\sum_{i=1}^{N}\beta_i\xi_i+\sum_{i=1}^{N}\beta_i-\sum_{i=1}^{N}\beta_iy_ib-(\sum_{i=1}^{N}\beta_iy_ix_i)^T(\sum_{j=1}^{N}\beta_hy_hx_j)\\\\
&s.t.\ 0\le\beta\le C,\ \sum_{i=1}^{N}\beta_iy_i=0\\\\
=&-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\beta_i\beta_jy_iy_jx_i^Tx_j+\sum_{i=1}^{N}\beta_i\\\\
\\\\
\Rightarrow\ &\underset{\beta}{\max} \Big[ -\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\beta_i\beta_jy_iy_jx_i^Tx_j+\sum_{i=1}^{N}\beta_i\Big]\\\\
\Rightarrow\ &\underset{\beta}{\min} \Big[\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\beta_i\beta_jy_iy_jx_i^Tx_j-\sum_{i=1}^{N}\beta_i\Big]\\\\
& s.t.\ 0\le \beta_i \le C,\ \sum_{i=1}^{N}\beta_iy_i=0 
\end{align*}
$$

#### **2.2. Feasible region:**

**Remember 3 constraints:**
* $\xi_i\ge 0$
* $0\le \beta_i \le C$
* $\sum\beta_iy_i=0$

#### **2.3. Complementary Slackness:** 

$\alpha_i^{\star}g_i(x)=0$ has 2 terms:

a.  $\alpha_i\xi_i=0 \Rightarrow\ (C-\beta_i)\xi_i=0$

b.  $\beta_i[y_i(w^Tx_i+b)+\xi_i-1]=0$



* Case 1.

    When $\beta_i>0$, 
    $$
    y_i(w^Tx_i+b)+\xi_i-1=0\\\\
    w^Tx_i+b=y_i-y_i\xi_i\\\\
    b^{\star}=y_i-y_i\xi_i-w^{\star T}x_i
    $$

* Case 2.

    When $\beta_i=0$, 
    $$\begin{align*}
    &y_i(w^Tx_i+b)+\xi_i-1\neq 0\ holds\\\\
    because\ &(C-\beta_i)\xi_i=0, \ therefore\  \xi_i = 0\\\\
    because\ &w^{\star}=\sum_{i=1}^{N}\beta_ix_iy_i, \ therefore\ w^{\star} = 0\\\\
    & b\ has\ infinite\ solutions\\\\
    \end{align*}
    $$
    
**Based on Case 1 and Case 2:**

When $0<\beta_i<C$,

$$
C-\beta_i>0 \Rightarrow\ \xi_i =0\\\\
b^{\star}=y_i-w^{\star T}x_i
$$

When $\beta_i=C$,

$$
C-\beta_i=0 \Rightarrow\ \xi_i\ge 0\\\\
b\ has\ infinite\ solutions\\\\
$$

#### **2.4. $\beta$ ranges:**

The range of $beta_i$ should be $0<\beta_i<C$

$$\begin{align*}
b^{\star}=y_i-w^{\star T}x_i,\ s.t.\ 0<\beta_i<C
\end{align*}
$$

where:

$$   
w^{\star}=\sum_{i=1}^{N}x_iy_i\beta_i\\\\
b^{\star}=y_i-w^{\star T}x_i\\\\
s.t.\ 0<\beta<C
$$

**In conclusion, we can only choose support vectors/points where $0<\beta_i<C$**.

#### 3. Conclusion

Base on Lagrange duality and KKT conditions, now we get the new target:

$$
\underset{\beta}{\min}: \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\beta_i\beta_jy_iy_jx_i^Tx_j-\sum_{i=1}^{N}\beta_i \\\\
s.t.  \Big\lbrace\begin{matrix}
0< \beta_i<C\\\\
\sum_{i=1}^{N}\beta_iy_i=0
\end{matrix}\
\\
$$

Optimal Solutions:

$W^{\star} = \sum_{i=1}^{N}x_iy_i\beta_i$

$b^{\star} = y_i-w^{\star T}x_i$


## SMO: Sequential Minimal Optimization

#### 1. A problem arises.

From solving duality problem, we get:

$$
\underset{\beta}{\min}: \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\beta_i\beta_jy_iy_jX_i^TX_j-\sum_{i=1}^{N}\beta_i \\\\
s.t.  \Big\lbrace\begin{matrix}
0< \beta_i<C\\\\
\sum_{i=1}^{N}\beta_iy_i=0
\end{matrix}\
\\
$$

there are N $\beta_i$. We need to find the best set of solutions.

#### 2. Key concepts

We can choose using **gradient descent** to find the best $\beta$。 But due to the constraint that $\sum\beta_iy_i=0$，we have to update paired gradients ($\beta_1,\ \beta_2$) simultaneously instead of updating only one.

#### 3. Algorithm

Let:

$$
W = \underset{\beta}{\min} \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\beta_i\beta_jy_iy_jX_i^TX_j-\sum_{i=1}^{N}\beta_i
$$

Every time, we only update tow $\beta$s, the rest of them were treated as constant **Const**.

Now let: $K_{ij} = x_i^Tx_j$

$$\begin{align*}
W(\beta_1,\beta_2) =  \underset{\beta1,\beta_2}{\min}\ & \frac{1}{2}\beta_1^2K_{11}+\frac{1}{2}\beta_2^2K_{22}+\beta_1\beta_2K_{11}\\\\ & +\beta_1y_1\sum_{j=3}^{N}\beta_jy_jK_{1j}+\beta_2y_2\sum_{j=3}^{N}\beta_jy_jK_{2j}-\beta_1-\beta_2-\sum_{i=3}^{N}\beta_i\\\\
s.t.\ &\beta_1y_1+\beta_2y_2=\sum_{i=1}^{N}\beta_iy_i - \sum_{i=3}^{N}\beta_iy_i=0-\sum_{i=3}^{N}\beta_iy_i = c\ (A\ constant)\\\\
& \beta_1,\beta_2\ge0\\\\\\
&Because\ y\in\lbrace+1,-1\rbrace,\ \beta_1y_1
+\beta_2y_2=c\\\\
& Therefore\ \beta_1=y_1(c-\beta_2y_2)\\\\
\Rightarrow  W(\beta_2)=\underset{\beta_2}{\min}\ & \frac{1}{2}y_1^2(c-y_2\beta_2)^2K_{11}+\frac{1}{2}\beta_2^2K_{22}\\\\
&+y_1(c-\beta_2y_2)\beta_2y_1y_2K_{12}\\\\
&+y_1^2(c-\beta_2y_2)\sum_{j=3}^{N}\beta_jy_jK_{1j}\\\\
&+\beta_2y_2\sum_{j=3}^{N}\beta_jy_jK_{2j}\\\\
&-y_1(c-\beta_2y_2)\\\\
&-\beta_2\\\\
&-\sum_{i=3}^{N}\beta_i
\end{align*}
$$

* **The derivative of $\beta_2$:**

$$\begin{align*}
&\frac{\partial W(\beta_2)}{\partial \beta_2}\\\\
=\ &-y_2(c-y_2\beta_2)K_{11}+\beta_2K_{22}+cy_2K_{12}-2\beta_2K_{12}-y_2\sum_{j=3}^{N}\beta_jy_jK_{1j}\\\\
\ &+y_2\sum_{j=3}^{N}\beta_jy_jK_{2j}+y_1y_2-y_2^2\\\\
=\ &0\\\\
\Rightarrow\ &\beta_2^{new}(K_{11}+K_{22}-2K_{12})\\\\
&=y_2^2-y_1y_2+cy_2K_{11}-cy_2K_{12}+y_2\sum_{j=3}^{N}\beta_j^{old}y_jK_{1j}-y_2\sum_{j=3}^{N}\beta_j^{old}y_jK_{2j}\\\\
&=y_2(y_2-y_1+cK_{11}-cK_{12}+\sum_{j=3}^{N}\beta_j^{old}y_jK_{1j}-\sum_{j=3}^{N}\beta_j^{old}y_jK_{2j})\\\\
&= y_2(y_2-y_1+cK_{11}-cK_{12}+f(x_1)-\beta_1^{old}y_1K_{11}-\beta_2^{old}y_2K_{12}-f(x_2)+\beta_1^{old}y_1K_{12}+\beta_2^{old}y_2K_{22})\\\\
\Rightarrow\ &\beta_2^{new} = \frac{y_2(y_2-y_1+cK_{11}-cK_{12}+f(x_1)-\beta_1^{old}y_1K_{11}-\beta_2^{old}y_2K_{12}-f(x_2)+\beta_1^{old}y_1K_{12}+\beta_2^{old}y_2K_{22})}{K_{11}+K_{22}-2K_{12}}
\end{align*}
$$

**Note:** $K_{12} = K_{21}$

-----

Since $f(x)=w^Tx+b$ and $w^{\star}=\sum_{j=1}^{N}\beta_jx_jy_j$, so

$$f(x_k)=\sum_{j=1}^{N}\beta_jy_jx_j^Tx_k+b$$,

Therefore,

$$
f(x_1)=\beta_1y_1K_{11}+\beta_2y_2K_{12}+\sum_{j=3}^{N}\beta_jy_jK_{1j}+b\\\\
f(x_2)=\beta_1y_1K_{21}+\beta_2y_2K_{22}+\sum_{j=3}^{N}\beta_jy_jK_{2j}+b
$$



-----

Given that 

$$\beta_1y_2+\beta_2y_2 = c,$$

we have: 

$$\beta_1^{new}y_1+\beta_2^{new}y_2=\beta_1^{old}y_1+\beta_2^{old}y_2=c,$$

So that we can replace $c$ with $\beta_1^{old}y_1+\beta_2^{old}y_2$,

$$\begin{align*}
\beta_2^{new} = &\frac{y_2(y_2-y_1+cK_{11}-cK_{12}+f(x_1)-\beta_1^{old}y_1K_{11}-\beta_2^{old}y_2K_{12}-f(x_2)+\beta_1^{old}y_1K_{12}+\beta_2^{old}y_2K_{22})}{K_{11}+K_{22}-2K_{12}}\\\\
=&\frac{y_2(y_2-y_1+\beta_1^{old}y_1K_{11}+\beta_2^{old}y_2K_{11}-\beta_1^{old}y_1K_{12}-\beta_2^{old}y_2K_{12}+f(x_1)\\\\-\beta_1^{old}y_1K_{11}-\beta_2^{old}y_2K_{12}-f(x_2)+\beta_1^{old}y_1K_{12}+\beta_2^{old}y_2K_{22})}{K_{11}+K_{22}-2K_{12}}\\\\
=&\frac{y_2\big[f(x_1)-y_1-(f(x_2)-y_2)+\beta_2^{old}y_2(K_{11}+K{22}-2K_{12})) \big]}{K_{11}+K_{22}-2K_{12}}
\end{align*}
$$

----

**OMG, almost there.**

----

Now we define Error term:

$$
Error:\ E_i = f(x_i)-y_i
$$

And set $\xi$ as:

$$
\xi = K_{11}+K_{22}-2K_{12}
$$

or more general: 

$$\xi=K_{mm}+K_{nn}-2K_{mn}$$


Then rewrite $\beta_2$ equation as:

$$\begin{align*}
\beta_2^{new} = \beta_2^{old}+\frac{y_2(E_1-E_2)}{\xi }
\end{align*}
$$

And 

$$\begin{align*}
\beta_1^{new}=& y_1(c-\beta_2^{new}y_2)\\\\
=&y_1(\beta_1^{old}y_1+\beta_2^{old}y_2-\beta_2^{new}y_2)\\\\
=&\beta_1^{old}+y_1y_2(\beta_2^{old}-\beta_2^{new})
\end{align*}
$$

#### When the decreasing rate of $W(\beta_k)$ is less than a threshold $T$, 

$$\begin{align*}
&\frac{W(\beta^{t})-W(\beta^{t+1})}{W(\beta^{t})}<T,\\\\
&Stop.
\end{align*}
$$




Whew, I made it. Thanks to COVID-19.
----


1. $\alpha_2^{new} = \alpha_2^{old} + y_2\frac{(E_1-E_2)}{\xi}$
2. $\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})$




