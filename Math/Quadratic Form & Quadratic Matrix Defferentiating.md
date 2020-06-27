# Quadratic Form & Quadratic Matrix Defferentiating

## 1. Quadratic Form

> Check 
> * [Diagonalization and Power of A](https://github.com/uttgeorge/Linear-Algebra/blob/master/22-Diagonalization%20and%20Power%20of%20A.pdf)
> * [Symmetric Matrices & Positive Definitness](https://github.com/uttgeorge/Linear-Algebra/blob/master/25-Symmetric%20Matrices%20%26%20Positive%20Definitness.pdf)
> * [Positive Difinite & Minima](https://github.com/uttgeorge/Linear-Algebra/blob/master/27-Positive%20Difinite%20%26%20Minima.pdf)
> * [Similar Matrices & Jordan Form](https://github.com/uttgeorge/Linear-Algebra/blob/master/28-Similar%20Matrices%20%26%20Jordan%20Form.pdf)

<!--### 1.1 Why only quadratic terms?

For a quadratic function, linear terms and constant can only shift or twist the polynomial with pure quadratic terms. -->

> ### 1.1 Matrix Form

> A quadratic form is a polynomial with terms all of degree two. Such as,
>  
>  $$2x_1^2+4x_1x_2+3x_2^2$$
>  
>  Any quadratic form can be expressed in matrix form as:
>  
>  $$X^TAX$$
>  
>  where $A$ is a symmetric matrix.
>  
>  For example, 
>  $$x_1^2+x_1x_2+x_2^2=1\\\\
>  \Rightarrow \begin{bmatrix}
>  x_1& x_2 
> \end{bmatrix}\begin{bmatrix}
>  1& 0.5\\\\ 
>  0.5& 1
> \end{bmatrix} \begin{bmatrix}
> x_1\\\\x_2 
> \end{bmatrix}=1$$

> ### 1.2 Diagonal Quadratic Form

> Diagonal Quadratic Form is not unique:

> **1. Eigendecomposition：**
> Since $A$ is a symmetric matrix, 
> $$A=Q\wedge Q^{-1}=Q\wedge Q^T$$
> where $Q$ is an orthogonal eigenvector matrix, and $\wedge$ is a diagonal eigenvalue matrix.

> Then,
> $$X^TAX=X^TQ\wedge Q^{T}X=(Q^TX)^T \wedge (Q^TX)=Y^T \wedge Y$$

> For example:
> $$
> A=\begin{bmatrix}
>  1& 0.5\\\\ 
>  0.5& 1
> \end{bmatrix}=\begin{bmatrix}
>  \frac{1}{\sqrt 2}& \frac{1}{\sqrt 2}\\\\ 
>  -\frac{1}{\sqrt 2}& \frac{1}{\sqrt 2}
> \end{bmatrix}\begin{bmatrix}
>  0.5& 0\\\\ 
>  0& 1.5
> \end{bmatrix}\begin{bmatrix}
>  \frac{1}{\sqrt 2}& -\frac{1}{\sqrt 2}\\\\ 
>  \frac{1}{\sqrt 2}& \frac{1}{\sqrt 2}
> \end{bmatrix}
> $$

> $$\begin{align*}\ &x_1^2+x_1x_2+x_2^2=1\\\\
> \Rightarrow\ &\begin{bmatrix}
> x_1& x_2 
> \end{bmatrix}\begin{bmatrix}
>  \frac{1}{\sqrt 2}& \frac{1}{\sqrt 2}\\\\ 
>  -\frac{1}{\sqrt 2}& \frac{1}{\sqrt 2}
> \end{bmatrix}\begin{bmatrix}
>  0.5& 0\\\\ 
>  0& 1.5
> \end{bmatrix}\begin{bmatrix}
>  \frac{1}{\sqrt 2}& -\frac{1}{\sqrt 2}\\\\ 
>  \frac{1}{\sqrt 2}& \frac{1}{\sqrt 2}
> \end{bmatrix} \begin{bmatrix}
> x_1\\\\x_2 
> \end{bmatrix}\\\\
> =\ & \begin{bmatrix}
>  \frac{1}{\sqrt 2}x_1- \frac{1}{\sqrt 2}x_2 &
>  \frac{1}{\sqrt 2}x_1+ \frac{1}{\sqrt 2}x_2
> \end{bmatrix}\begin{bmatrix}
>  0.5& 0\\\\ 
>  0& 1.5
> \end{bmatrix}\begin{bmatrix}
>  \frac{1}{\sqrt 2}x_1 -\frac{1}{\sqrt 2}x_2\\\\ 
>  \frac{1}{\sqrt 2}x_1+ \frac{1}{\sqrt 2}x_2
> \end{bmatrix}\\\\
> =\ & 0.5(\frac{1}{\sqrt 2}x_1 -\frac{1}{\sqrt 2}x_2)^2+1.5(\frac{1}{\sqrt 2}x_1+ \frac{1}{\sqrt 2}x_2)^2\\\\
> =\ & 0.5y_1^2+1.5y_2^2
> \end{align*}$$

> $$\begin{align*}
> \left \lbrace\begin{matrix}
> y_1=\frac{1}{\sqrt 2}x_1 -\frac{1}{\sqrt 2}x_2
> \\\\
> y_2=\frac{1}{\sqrt 2}x_1+ \frac{1}{\sqrt 2}x_2
> \end{matrix}\right.\Rightarrow \left \lbrace\begin{matrix}
> x_1=\frac{1}{\sqrt 2}(y_1+y_2)
> \\\\
> x_2=\frac{1}{\sqrt 2}(y_2-y_1)
> \end{matrix}\right.
> \end{align*}$$


> **2. Completing the square：**
> $$A=LU$$

> For example:
> $$
> A=\begin{bmatrix}
>  1& 0.5\\\\ 
>  0.5& 1
> \end{bmatrix}=\begin{bmatrix}
>  1& 0\\\\ 
>  0.5& 1
> \end{bmatrix}\begin{bmatrix}
>  1& 0.5\\\\ 
>  0& 3/4
> \end{bmatrix}
> $$


> $$\begin{align*}&\ x_1^2+x_1x_2+x_2^2=1\\\\
> \Rightarrow\ &\begin{bmatrix}
> x_1& x_2 
> \end{bmatrix}\begin{bmatrix}
>  1& 0\\\\ 
>  0.5& 1
> \end{bmatrix}\begin{bmatrix}
>  1& 0.5\\\\ 
>  0& 3/4
> \end{bmatrix} \begin{bmatrix}
> x_1\\\\x_2 
> \end{bmatrix}\\\\
> =\ & \begin{bmatrix}
> x_1+0.5x_2& x_2 
> \end{bmatrix} \begin{bmatrix}
> x_1+0.5x_2\\\\(3/4) x_2 
> \end{bmatrix}\\\\\
> =\ & (x_1+0.5x_2)^2+(3/4) x_2 ^2\\\\
> =\ &\begin{bmatrix}
> x_1+0.5y& x_2 
> \end{bmatrix} \begin{bmatrix}
>  1& 0\\\\ 
>  0& 3/4
> \end{bmatrix}\begin{bmatrix}
> x_1+0.5x_2\\\\ x_2 
> \end{bmatrix}\\\\
> =\ & y_1^2+(3/4) y_2 ^2\end{align*}$$

> $$\begin{align*}
> \left \lbrace\begin{matrix}
> y_1=x_1+0.5x_2
> \\\\
> y_2=x_2
> \end{matrix}\right.\Rightarrow \left \lbrace\begin{matrix}
> x_1=y_1-0.5y_2
> \\\\
> x_2=y_2
> \end{matrix}\right.
> \end{align*}$$

> ### 1.3 Normalized Quadratic Form

> Every quadratic form can be reducible to the form:
> $$
> x_1^2+x_2^2+...+x_p^2-x_{p+1}^2-...-x_{r}^2+0x_{r+1}...
> $$

> Here $r$ is the rank of A, the number p (the number of positive elements) and r−p (the number of negative elements) are uniquely determined by A based on Sylvester's law of inertia. Thus, the number of negative elements in the diagonal of A is always the same; and the same goes for the number of positive elements. 

> Any quadratic form has a unique normalized form.

> ### 1.4 Positive Definite Matrix

> How to tell if a matrix is positive definite or not?

> For $f(x) = X^TAX$:
> * $f(x)>0,\ x\ne0,\ x\in\mathbb{R}$, $A$ is positive definite;
> * $f(x)\ge0,\ x\ne0,\ x\in\mathbb{R}$, $A$ is positive semi-definite;
> * $f(x)<0,\ x\ne0,\ x\in\mathbb{R}$, $A$ is negative definite;
> * $f(x)\le 0,\ x\ne0,\ x\in\mathbb{R}$, $A$ is negative semi-definite;
> * or indefinite.

> If eigenvalues of $A$ are greater than 0, then $X^TAX$ is positive definite.

## 2. Derivative of Quadratic Form

For any quadratic form $X^TAX$, the derivative is $X^T(A+A^T)$

 Proof:
 $$X^TAX=\sum_{j=1}^{N}\sum_{i=1}^{N}a_{i,j}x_{i}x_{j}$$
 Differentiating with respect to the $k_{th}$ element of x we have
 $$
 \frac{\partial (X^TAX)}{\partial x_k}=\sum_{j=1}^{N}a_{k,j}x_{j}+\sum_{i=1}^{N}a_{i,k}x_{i}
 $$
 for all k = 1, 2, . . . , n, and consequently,
 $$
  \frac{\partial (X^TAX)}{\partial X}=X^TA^T+X^TA=X^T(A^T+A)
 $$
 q.e.d.
 
## References

* https://encyclopediaofmath.org/wiki/Quadratic_form
* https://atmos.uw.edu/~dennis/MatrixCalculus.pdf