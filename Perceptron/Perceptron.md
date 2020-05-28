# Linear-Model
Combination of Linear Models

## 1.Perceptron

* Idea: Driven by mistakes
* Model:

\begin{align}
f(x)=sign(w^Tx+b),\\x{\in}R^p,w{\in}R^p\\\\
sign(a) = \left\{\begin{matrix}
+1,a\geqslant0
\\ 
-1,a<0
\end{matrix}\right.
\end{align}

* Loss funtion:

    1. Use the number of missclassification as loss

 \begin{align}
L(w)=\sum_{i=1}^{N}I\left \{y_i(w^Tx_i+b)<0\right \}\\
\left.\begin{matrix}
w^Tx_i+b>0,y_i>0
\\ 
w^Tx_i+b<0,y_i<0
\end{matrix}\right\}\Rightarrow \left\{\begin{matrix}
w^Tx_i+b>0, True
\\ 
w^Tx_i+b<0, False
\end{matrix}\right.
\end{align}

    **But in this case, the funtion is not derivative.**
    
    2. Use the distance as loss

 \begin{align}
min:L(w)=\sum_{x_i{\in}D}^{}-y_i(w^Tx_i+b)\\D:\left \{ Miss\ Classified\ Points\right \}\\
\Delta _{w}L = \sum_{}^{} -y_ix_i\\
\Delta _{b}L = \sum_{}^{} -y_i
\end{align}

* Algorithm: **SGD**

 \begin{align}
w^{(t+1)}=w^{(t)}-\lambda\Delta _{w}L \\
=w^{(t)} + \lambda\sum_{}^{} y_ix_i\\\\
b^{(t+1)}=b^{(t)}\lambda\Delta _{b}L \\
=b^{(t)} + \lambda\sum_{}^{} y_i
\end{align}

    b could be treated as w0
