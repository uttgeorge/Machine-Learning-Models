# SVM

**Please check the .ipynb files instead of readme files**


SVM code from scratch
**Classification**

Prerequisite：Linearly separable

Basic idea is Max Margin Classifier, we have to find the widest road between class1 and class2.

\begin{align}
max:margin(w,b)
\\st.
\left\{\begin{matrix}
w^Tx_i +b>0,y_i=+1\\ 
w^Tx_i +b<0,y_i=-1
\end{matrix}\right.
\\
 \Rightarrow y_i(w^{T}x_{i}+b) > 0\\
 \forall i=1,2,...,N
\end{align}

1. For the hard margin:

   * We set margin equals 1, and $y_i(w^Tx_i+b)\geqslant1$;
    
2. For the soft margin:

   * We allow some noise, so that we can increase the robustness of our system. 
   
   * We introduced slckness variable $\xi$, which represent **loss**. 
   
   * And now the margin function changes to $y_i(w^Tx_i+b)\geqslant1-\xi_i,s.t.xi\geqslant0$

Suppose we have two classes X and O, for class X:

   1. When X is correctly classified, means X locates outside the margin, then $\xi=0$
   2. WHen X is incorrecly classified and locates on the right side of $y_i(w^Tx_i+b)\geqslant0$,then $0<\xi\leqslant0$ 
   3. When X is on the other side of margin, which means $y_i(w^Tx_i+b)\leqslant0$, then $\xi>1$
   
Based on these 3 conditions, we get the new target funtion:

\begin{align}
min_{w,b,\xi }:\frac{1}{2}\left \| w \right \|^{2}+C\sum_{i=1}^{N}\xi_i\\
s.t. \left\{\begin{matrix}y_i(w^Tx_i + b)\geqslant 1-\xi_i
\\ \xi_i\geqslant0\\
\end{matrix}\right.
\\
\end{align}

The loss function here called **Hinge Loss**, basically it uses distance to measure **loss**: $\xi$ represents the distance from a point to its corresponding margin $w^Tx+b=1$ when it is miss-classified.

   1. If $w^Tx+b\geqslant1$, $\xi_i=0$, No loss, correct.
   2. If $w^Tx+b<1$, $\xi_i=1-y_i(w^Tx+b)$
   
So now we have:
   \begin{align}
\xi_i =max\left \{ 0,1-y_i(w^Tx_i + b) \right \}
\end{align}
   
Base on lagrange duality and KKT conditions, now we get the new target:



\begin{align}
min: \sum_{i=1}^{N}\alpha_i - \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_jX_i^TX_j\\
s.t.  \left\{\begin{matrix}
0< \alpha_i<C\\
\sum_{i=1}^{N}\alpha_iy_i=0
\end{matrix}\right.
\\
\end{align}

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



