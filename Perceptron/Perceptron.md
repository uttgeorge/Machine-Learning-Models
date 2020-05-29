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

    1. Use the number of missclassification as loss

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

    **But in this case, the funtion is not derivative.**
    
    2. Use the distance as loss

$$min:L(w)=\sum_{x_i{\in}D}^{}-y_i(w^Tx_i+b)\\\\D:\lbrace Miss\ Classified\ Points\rbrace\\\\
\Delta _{w}L = \sum_{}^{} -y_ix_i\\\\
\Delta _{b}L = \sum_{}^{} -y_i$$

* Algorithm: **SGD**

$$w^{(t+1)}=w^{(t)}-\lambda\Delta _{w}L \\
=w^{(t)} + \lambda\sum_{}^{} y_ix_i\\\\
b^{(t+1)}=b^{(t)}\lambda\Delta _{b}L \\
=b^{(t)} + \lambda\sum_{}^{} y_i$$

    b could be treated as w0
