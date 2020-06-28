# Newton's method & Quasi-Newton method （BFGS ongoing）

## 1. Newton's method

Similar to the **Gradient Descent**, Newton's method is an iterative approach to find the point where the derivative equals 0. 

The key point of Newton's method is using quadratic function including 1st derivative and 2nd derivative to approximate the target function.

### _**1.1 One Dimensional (one variable) function**_
The Taylor polynomial of target function at point $x_0$:
$$
f(x)\approx f(x_0)+f'(x_0)(x-x_0)+\frac{1}{2}f''(x_0)(x-x_0)^2
$$
Now, find the point where the derivative equals 0.
$$
f'(x)=f'(x_0)+(x-x_0)f''(x_0)=0
$$
Solve the equation and we get:
$$
x=x_0-\frac{f'(x_0)}{f''(x_0)}
$$
### _**1.2 High-Dimensional (Multivariate) function**_
Again, the Taylor polynomial of target function at point $x_0$:
$$
f(X+\Delta X)\approx f(X)+(\Delta X)^T \nabla f(X)+\frac{1}{2}(\Delta X)^T\nabla^2 f(X)(\Delta X)
$$
$$
f(X)\approx f(X_0)+\nabla f(X_0)^T(X-X_0)+\frac{1}{2}(X-X_0)^T\nabla^2 f(X_0)(X-X_0)
$$
Now, find the point where the derivative equals 0.
$$
\nabla f(X)=\nabla f(X_0)+\nabla^2 f(X_0)(X-X_0)=0
$$
Where $\nabla^2 f(X_0)$ is a [Hessian Matrix](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/Math/Jacobian%20%26%20Hessian%20Matrix.md), denote as $H$. Solve the equation and we get:
$$\begin{align*}
X = X_0 - (\nabla^2 f(X_0))^{-1}(\nabla f(X_0))
\end{align*}
$$
Denote gradient as $g$, then
$$
X = X_0 - H^{-1}g
$$

**This is a set of solutions for a set of linear functions.**

Start from the beginning point $X_0$, iterate the process
$$
X_{k+1}=X_{k} - H_k^{-1}g_k
$$

### 1.3 Line Search

### 1.4 Pitfalls of Newton's method

## 2. Quasi-Newton method

Newton's method requires computing Hessian Matrix every iteration, then computing a set of equations based on Hessian Matrix. Besides that, Hessian Matrix may not invertible. 

A revised approach is Quasi-Newton method. It does not compute Hessian Matrix and its inverse, but builds a symmetric positive-definite matrix that approximate Hessian Matrix. 

The Taylor polynomial at $x_{k+1}$:


$$
f(X)\approx f(X_{k+1})+\nabla f(X_{k+1})^T(X-X_{k+1})+\frac{1}{2}(X-X_{k+1})^T\nabla^2 f(X_{k+1})(X-X_{k+1})
$$

$$
\nabla f(X)\approx \nabla f(X_{k+1})+\nabla^2 f(X_{k+1})(X-X_{k+1})
$$
Set $X=X_k$:
$$
\nabla f(X_k)\approx \nabla f(X_{k+1})+\nabla^2 f(X_{k+1})(X_k-X_{k+1})
$$

$$
\nabla f(X_{k+1})- \nabla f(X_k)\approx \nabla^2 f(X_{k+1})(X_{k+1}-X_k)
$$

$$
g_{k+1}- g_k\approx \nabla^2 f(X_{k+1})(X_{k+1}-X_k)
$$

If we set:

$$
s_k=X_{k+1}-X_k\\\\
y_k=g_{k+1}- g_k
$$

Then we get:

$$
y_k \approx H_{k+1}s_k\\\\
s_k\approx H_{k+1}^{-1} y_k
$$

And this is called quasi-newton condition.

#### **BFGS** ？？？？？？？？？？

Build a approximate matrix B of Hessian Matrix:

$$
B_k \approx H_k
$$

and iterative update it:

$$
B_{k+1}=B_k+\Delta B_k
$$

$B_0$ is an Identity Matrix. So the problem now is to solve $\Delta B_k$.

$$
\Delta B_k = \alpha uu^T+\beta vv^T
$$

where

$$\begin{align*}
u&=y_k\\\\
v&=B_ks_k\\\\
\alpha&=\frac{1}{y_k^Ts_k}\\\\
\beta&=-\frac{1}{s_k^TB_ks_k}
\end{align*}
$$

$$\begin{align*}
\Delta B_k=\frac{y_ky_k^T}{y_k^Ts_k}-\frac{B_ks_ks_k^TB_k}{s_k^TB_ks_k}
\end{align*}
$$

