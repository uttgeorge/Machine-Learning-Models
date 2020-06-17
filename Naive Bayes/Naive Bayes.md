# Naive Bayes
## 1. Background Knowledge

[Bayesian Estimation, MLE, MAP](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/Math/Bayesian%20Estimation%2C%20MLE%2C%20MAP.md)

$$\begin{align*}
P(\theta|X) &= \frac{P(X| \theta) \cdot P(\theta)}{\int_{\theta}P(X|\theta)P(\theta)d\theta}
\end{align*}
$$
where $X$ represents features and $\theta$ represents classes. 

## 2. Why Naive?

Naive Bayes assumes that all features are mutually independent.

$$
X_i \perp X_j,\ i \neq j
$$

So that, likelihood $P(X|\theta) = \prod_{i=1}^{N}P(x_i|\theta)$ 

## 3. How it works? Multinomial Naive Bayes.

Now, we have

$$\begin{align*}
P(\theta|x_1,x_2,..,x_N) &= \frac{P(\theta) \cdot \prod_{i=1}^{N}P(x_i|\theta)}{P(X)}
\end{align*}
$$

We can simply compare posterior probability of $\theta_j$ by calculating $P(\theta_j|x_1,x_2,..,x_N)$.  And since $P(X)$ is fixed, we have
$$
P(\theta_j|x_1,x_2,..,x_N) \propto P(\theta_j) \cdot \prod_{i=1}^{N}P(x_i|\theta_j)
$$


_**Example:**_
    
Apple Quality:

| ID | SIZE  | COLOR | SHAPE     | QUALITY |
|----|-------|-------|-----------|--------------|
| 1  | Small | Green | Irregular | Bad        |
| 2  | Big   | Red   | Irregular | Good         |
| 3  | Big   | Red   | Sphere    | Good         |
| 4  | Big   | Green | Sphere    | Bad        |
| 5  | Medium   | Green | Irregular | Bad        |
| 6  | Small | Red   | Sphere    | Good         |
| 7  | Big   | Green | Irregular | Bad        |
| 8  | Small | Red   | Irregular | Bad        |
| 9  | Small | Green | Sphere    | Bad        |
| 10 | Big   | Red   | Sphere    | Good         |
    
What is the quality of an apple if it is (Big, Red, Sphere)?

$$\begin{align*}
P(Good|Big, Red, Sphere) & \propto P(Good)P(Big|Good)P(Red|Good)P(Sphere|Good)\\\\
&=0.4\cdot\frac{3}{4}\cdot\frac{4}{4}\cdot\frac{3}{4}\\\\
&=0.225\\\\
P(Bad|Big, Red, Sphere) & \propto P(Bad)P(Big|Bad)P(Red|Bad)P(Sphere|Bad)\\\\
&=0.6\cdot\frac{2}{6}\cdot\frac{1}{6}\cdot\frac{2}{6}\\\\
&=0.0111\\
\end{align*}
$$

So, it a good apple.



## 4. Laplace Smoothing
In statistics, Laplace Smoothing is a technique to smooth categorical data. Laplace Smoothing is introduced to solve the problem of zero probability.

Suppose we want to measure the quality of an apple if it is (Medium, Red, Shpere)?

$$\begin{align*}
P(Good|Medium, Red, Sphere) & \propto P(Good)P(Medium|Good)P(Red|Good)P(Sphere|Good)\\\\
&=0.4\cdot\frac{0}{4}\cdot\frac{4}{4}\cdot\frac{3}{4}\\\\
&=0\\\\
P(Bad|Medium, Red, Sphere) & \propto P(Bad)P(Medium|Bad)P(Red|Bad)P(Sphere|Bad)\\\\
&=0.6\cdot\frac{1}{6}\cdot\frac{1}{6}\cdot\frac{2}{6}\\\\
&=0.006\\
\end{align*}
$$

The result does not make sense, because as we know, this apple is more likely to be a good one. But because of lacking medium size data in our dataset, we can not correctly classify.

To overcome this problem, we implement Laplace Smoothing.

**Before implement Laplace Smoothing, the Likelihood is:**
$$
P(x_i|\theta_j)=\frac{Count(x_i|\theta_j)}{Count(\theta_j)}
$$
or
$$
P(x_i|\theta_j)=\frac{\sum_{i=1}^{N}I(x_i,\theta_j)}{\sum_{i=1}^{N}I(\theta_j)}
$$
**After implement Laplace Smoothing, the Likelihood become:**
$$
P(x_i|\theta_j)=\frac{Count(x_i|\theta_j)+\alpha}{Count(\theta_j)+\alpha|V|}
$$
or
$$
P(x_i|\theta_j)=\frac{\sum_{i=1}^{N}I(x_i,\theta_j)+\alpha}{\sum_{i=1}^{N}I(\theta_j)+\alpha|V|}
$$
where $|V|$ is the number of different categories in $x_i$, and $\alpha$ is a smoothing parameter and $\alpha>0$.

When $\alpha=0$, there is no smoothing. And normally we set $\alpha=1$

## 5. Underflow

Since probabilities are always between 0 and 1, the product of many probabilities may ends up to 0. This is called **underflow**. In order to solve this problem, we simply calculate the sum of log of probabilities instead of product of probabilities.

$$\begin{align*}
&\prod_{i=1}^{N}P(x_i|\theta)\\\\
\Rightarrow &\sum_{i=1}^{N}\log P(x_i|\theta)
\end{align*}
$$

## 6. Gaussian Naive Bayes/ Bernoulli Naive Bayes

* **Gaussian Naive Bayes:**

    Gaussian NB deals with numerical values, instead of counting each feature and class, GNB find the mean and variance to calculate likelihood.

* **Bernoulli Naive Bayes:**

    Similar to Multinomial NB, but only process binary values.

## Pro & Con

### Pro
1. Super easy
2. Not very sensitive to the missing data
3. Low variance: model is too simple.

### Con
1. **Strict independent assumption.** If features are highly correlated, naive bayes will ignore that and lead to poor result.
2. High Bias: model is too simple.


## References

[Additive smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)

[理解朴素贝叶斯分类的拉普拉斯平滑](https://zhuanlan.zhihu.com/p/26329951)
