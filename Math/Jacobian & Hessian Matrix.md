# Jacobian & Hessian Matrix

## 1. Jacobian 

### 1.1 Jacobian Matrix

The Jacobian matrix of a vector-valued function in several variables is the matrix of all its first-order partial derivatives. 

Suppose $f: \mathbb{R}_n \rightarrow \mathbb{R}_m$, is a vector-valued function that each of its first-order partial derivatives exist on $\mathbb{R}_n$. This function takes a point $x\in \mathbb{R}_n$ as input and produces the vector $f(x)\in \mathbb{R}_m$ as output. To be more clear, $f$ is a vector-valued function of several real-valued functions.


$$
\left\lbrace\begin{matrix}
y_1=f_1(x_1,x_2,...,x_n)
\\\\
y_2=f_2(x_1,x_2,...,x_n)
\\\\
... 
\\\\
y_m=f_m(x_1,x_2,...,x_n) 
\end{matrix}\right.
$$

Then the Jacobian matrix of $f$ is defined to be an $m\times n$ matrix, denoted by $J$, whose $(i,j)_th$ entry is $J_{i,j}=\frac{\partial {f_i}}{\partial {x_j}}$:

$$
J=[\frac{\partial {f}}{\partial {x_1}} ... \frac{\partial {f}}{\partial {x_n}}]= \begin{bmatrix}
\frac{\partial {f_1}}{\partial {x_1}} & ... & \frac{\partial {f_1}}{\partial {x_n}}\\\\
. & . & .\\\\ 
\frac{\partial {f_m}}{\partial {x_1}} & ... & \frac{\partial {f_m}}{\partial {x_n}}
\end{bmatrix}
$$

When $m=1$, that is when $f: \mathbb{R}_n → \mathbb{R}$ is a scalar-valued function, the Jacobian matrix reduces to a row vector. This row vector of all first-order partial derivatives of $f$ is the transpose of the gradient of $f$. Specialising further, when $m = n = 1$, that is when $f: \mathbb{R} → \mathbb{R}$ is a scalar-valued function of a single variable, the Jacobian matrix has a single entry. This entry is the derivative of the function $f$.


If $f$ is differentiable at a point $p$ in $\mathbb{R}_n$, then its differential is represented by $J_{f(p)}$. In this case, the linear transformation represented by $J_{f(p)}$ is the best linear approximation of $f$ near the point $p$, in the sense that

$$f(x)\approx f(p) + J_f(p)\cdot (x-p)$$

### 1.2 Jacobian Determinant

When $m = n$, the Jacobian matrix is square, so its determinant is a well-defined function of $x$, known as the Jacobian determinant of $f$. It carries important information about the local behavior of $f$. 

1. $det(J) \neq 0$:
$f$ is invertible near a point $p \in \mathbb{R}_n$ if and only if the Jacobian determinant at $p$ is non-zero

2. $det(J) > 0$:
$f$ preserves orientation near $p$

3. $det(J) < 0$:
$f$ reverses orientation

## 2. Hessian Matrix

The Hessian matrix or Hessian is a square matrix of second-order partial derivatives of a scalar-valued function $f$, or scalar field. It describes the local curvature of a function of many variables.

$$f(x_1,x_2,...,x_n)$$

Suppose $f: \mathbb{R}_n → \mathbb{R}$ is a function taking as input a vector $x \in \mathbb{R}_n$ and outputting a scalar $f(x) \in \mathbb{R}$. If all second partial derivatives of $f$ exist and are continuous over the domain of the function, then the Hessian matrix $H$ of $f$ is a square $n\times n$ matrix, usually defined and arranged as follows:


$$
\begin{bmatrix}
\frac{\partial^2 {f}}{\partial {x^2_1}} & \frac{\partial^2 {f}}{\partial {x_1}\partial {x_2}} & ...& \frac{\partial^2 {f}}{\partial {x_1}\partial {x_n}}\\\\
\frac{\partial^2 {f}}{\partial {x_2}\partial {x_1}} & \frac{\partial^2 {f}}{\partial {x_{2}^2}} & ...& \frac{\partial^2 {f}}{\partial {x_2}\partial {x_n}} \\\\ 
... & ... & ...& ...\\\\ 
\frac{\partial^2 {f}}{\partial {x_n}\partial {x_1}} & \frac{\partial^2 {f}}{\partial {x_n}\partial {x_2}} & ... & \frac{\partial {f}}{\partial {x^2_{n}}}
\end{bmatrix}
$$

The Hessian matrix is a symmetric matrix.



## References 

* https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
* https://en.wikipedia.org/wiki/Hessian_matrix