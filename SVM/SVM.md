# SVM

**Please check the .ipynb files instead of readme files**

## Duality

#### 1. Primal Problem: 

$$\begin{align*}
\min:\ &f(x)\\\\
s.t.\ &g_i(x)\le0,i=1,2,...,k\\\\
&h_j(x)=0,j=1,2,...,l
\end{align*}
$$

Introduce **Generalized Lagrange Function(Find extrema under contraints):**

$$\begin{align*}
L(x,\alpha,\beta)&=f(x)+\sum_{i=1}^{k}\alpha_ig_i(x)+\sum_{j=1}^{l}\beta_jh_j(x)\\\\
s.t.\ &\alpha_i\ge 0
\end{align*}
$$


This function means: let $\alpha$ and $\beta$ be fixed, iterate every $x$ to find the minimum $L(x,\alpha,\beta)$.

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
& \underset{x}{\min} \underset{\alpha,\beta}{\max}L(x,\alpha,\beta) = \underset{x}{\min}(+\infty,f(x))=\underset{x}{min}f(x)\\\\
\end{align*}
$$

$Q.E.D$

#### 2. Dual Problem:

**Definition of duality problems:**

$$\begin{align*}
\max:\ \theta(\alpha,\beta)&=\underset{x\in \mathbb{R}^p}{inf} \big\lbrace L(x,\alpha,\beta) \big\rbrace\\\\
s.t.\ &\alpha_i\ge 0
\end{align*}
$$

**If $x^*$ is the solution of the primal problem, and $\alpha^*$, $\beta^*$ are the solutions of the dual problem, then $$f(x^*)\ge\theta(\alpha^*,\beta^*)$$**

$Proof:$

$$\begin{align*}
\theta(\alpha^*,\beta^*)&=\underset{x\in \mathbb{R}^p}{inf} \big\lbrace L(x,\alpha,\beta) \big\rbrace\\\\
&\le L(x,\alpha,\beta)\\\\
&= f(x^*)+\sum_{i=1}^{k}\alpha_i^*g_i(x^*)+\sum_{j=1}^{l}\beta_j^*h_j(x^*),\ s.t.\ \alpha^*\ge 0,\ g_i(x)\le 0, h_j(x)=0\\\\
&le f(x^*)
\end{align*}
$$

$Q.E.D$

This is so called: **Weak Duality.**

## KKT

#### 1. Definition 
$$G=f(x^*)-\theta(\alpha^*,\beta^*)\ge 0$$
where G is the <font color="red">**Duality Gap**</font> between primal problem and dual problem.
Under some condition, G=0. And this is called **strong duality.**

#### 2. Strong Duality Theorem
If $f(x)$ is a convex function, and $g(x)=A^Tx+b,\ h(x)=C^Tx+d$,
then the duality gap between primal and dual problem for this optimization problem is 0.
In other word:
$$
f(x^*)= \theta(\alpha^*,\beta^*)
$$

#### 3. KKT Condition (Karush-Kuhn-Tucker)

1. $L$ is derivative, and can find an optimal solution. (Convex)
    $
    \nabla_xL=0,\ \nabla_{\alpha}L=0,\ \nabla_{\beta}L=0
    $
    
1. $\alpha_i\ge 0$

2. $g_i(x) \le 0, h_j(x)=0$
3. $\alpha_i \cdot g_i(x) =0,i=1,2,...,k$

$$
in\ order\ to\ let:\ f(x^*)+\sum \alpha_ig_i(x)+0=f(x^*)\\\\
\because \alpha_i \ge 0\ and\ g_i(x) \le 0\\\\
\therefore \alpha_i g_i(x)=0
$$

<font color="red">**KKT condition if sufficient and necessary for strong duality!**</font>

## Hard Margin

**Classification**

Prerequisite：Linearly separable

Basic idea is Max Margin Classifier, we have to find the widest road between class1 and class2.

$$
max:margin(w,b)
\\\\
s.t.\Bigg\lbrace\begin{matrix}
w^Tx_i +b>0,y_i=+1\\\\
w^Tx_i +b<0,y_i=-1
\
\end{matrix}
\\\\
 \Rightarrow y_i(w^{T}x_{i}+b) > 0
 \\\\
 \forall i=1,2,...,N
$$

1. For the hard margin:

   * We set margin equals 1, and $y_i(w^Tx_i+b)\ge1$;
    
2. For the soft margin:

   * We allow some noise, so that we can increase the robustness of our system. 
   
   * We introduced slckness variable $\xi$, which represent **loss**. 
   
   * And now the margin function changes to $y_i(w^Tx_i+b)\ge1-\xi_i,s.t.xi\ge0$

Suppose we have two classes X and O, for class X:

   1. When X is correctly classified, means X locates outside the margin, then $\xi=0$
   2. When X is incorrecly classified and locates on the right side of $y_i(w^Tx_i+b)\ge0$,then $0<\xi\le0$ 
   3. When X is on the other side of margin, which means $y_i(w^Tx_i+b)\le0$, then $\xi>1$
   
Based on these 3 conditions, we get the new target funtion:

$$
min_{w,b,\xi }:\frac{1}{2}\left \| w \right \|^{2}+C\sum_{i=1}^{N}\xi_i\\\\
s.t. \Big\lbrace\begin{matrix}y_i(w^Tx_i + b)\ge 1-\xi_i
\\\\ \xi_i\ge0\\
\end{matrix}\
\\
$$

The loss function here called **Hinge Loss**, basically it uses distance to measure **loss**: $\xi$ represents the distance from a point to its corresponding margin $w^Tx+b=1$ when it is miss-classified.

   1. If $w^Tx+b\ge1$, $\xi_i=0$, No loss, correct.
   2. If $w^Tx+b<1$, $\xi_i=1-y_i(w^Tx+b)$
   
So now we have:
$$
\xi_i =max\lbrace 0,1-y_i(w^Tx_i + b) \rbrace
$$
   
Base on lagrange duality and KKT conditions, now we get the new target:



$$
min: \sum_{i=1}^{N}\alpha_i - \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_jX_i^TX_j\\\\
s.t.  \Big\lbrace\begin{matrix}
0< \alpha_i<C\\\\
\sum_{i=1}^{N}\alpha_iy_i=0
\end{matrix}\
\\
$$

Optimal Solutions:

$W^* = \sum_{i=1}^{N}x_iy_i\alpha_i$

$b^* = y_i-w^Tx_i$


#### SMO: Sequential Minimal Optimization

Basic idea: SGD and paired $\alpha$

Each time, we only update tow $\alpha$s, the rest of them were treated as constant **Const**.

Now set:

1. $K_{ij} = X_i^TX_j$
2. $f(x_k)=\sum_{i=1}^{N}\alpha_iy_iX_i^Tx_k+b$
3. Error: $E_i = f(x_i)-y_i$
4. $\xi=K_{mm}+K_{nn}-2K_{mn}$

Then we get:

1. $\alpha_2^{new} = \alpha_2^{old} + y_2\frac{(E_1-E_2)}{\xi}$
2. $\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})$



