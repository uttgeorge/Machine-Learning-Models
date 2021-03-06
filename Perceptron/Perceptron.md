# Linear-Model
Combination of Linear Models

## 1.Perceptron

* Idea: Driven by mistakes
* Model:


$$
f(x)=sign(w^Tx+b),
\\\\
x{\in}R^p,w{\in}R^p\\\\
sign(a) = \Big\lbrace\begin{matrix}
+1,a\ge0
\\\\
-1,a<0
\end{matrix}\
$$


* Loss funtion:

     **1. Use the number of missclassification as loss**
 $$
L(w)=\sum_{i=1}^{N}I\lbrace y_i(w^Tx_i+b)<0\rbrace\\\\
\begin{matrix}
w^Tx_i+b>0,y_i>0
\\\\
w^Tx_i+b<0,y_i<0
\end{matrix}\Big\rbrace\Rightarrow \Big\lbrace\begin{matrix}
w^Tx_i+b>0, True
\\\\
w^Tx_i+b<0, False
\end{matrix}\
$$

    But in this case, the funtion is not derivative.

     **2. Use the distance as loss**

$$
\begin{align*}
& min:L(w)=\sum_{x_i{\in}D}^{}-y_i(w^Tx_i+b)\\\\& D:\lbrace Miss\ Classified\ Points\rbrace\\\\
& \Delta_{w}L = \sum -y_ix_i\\\\
& \Delta_{b}L = \sum -y_i
\end{align*}
$$


* Algorithm: SGD
$$
\begin{align*}
w^{(t+1)} & =w^{(t)}-\lambda\Delta_{w}L\\\\
& =w^{(t)} + \lambda\sum y_ix_i\\\\
b^{(t+1)}&=b^{(t)}\lambda\Delta_{b}L\\\\
&=b^{(t)} + \lambda\sum y_i
\end{align*}
$$
$b$ could be treated as $w_0$



## 2. MLP

* Limitation
PLA has limitation, it can not solve **Nonlinear problem**, such as XOR problem.

* Universal Approximation theorem
$\le 1$ hidden layer can approximate any function

    if 1 hidden layer is good, why deep? 
    And backpropagation can lead to vanishing gradient problem


 
