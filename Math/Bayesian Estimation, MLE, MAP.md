# Bayesian Estimation, MLE, MAP

## **1. Overview**

Basic concepts of Bayesian Estimation, Maximum Likelihood Estimation, Maximum A Posteriori Estimation

## **2. Background Knowledge**

**2.1 Probability and Statistics** 
* **Probability:** **Given a data generating process, what are the properties of the outcomes?**

    _概率论是在给定条件（已知模型和参数）下，对要发生的事件（新输入数据）的预测。_
* **Statistical Inference:** **Given the outcomes, what can we say about the process that generated the data**

    _统计推断是在给定数据（训练数据）下，对数据生成方式（模型和参数）的归纳总结。_
    
    
**2.2 Descriptive statistics & Statistical inference**
* **Descriptive statistics:** are brief descriptive coefficients (eg. _mean, variance, median, quartile_ ) that summarize a given data set, which can be either a representation of the **entire** or **a sample of a population**.
* **Statistical inference:** is the process of using data analysis from   **a sample of a population** to deduce properties of an underlying distribution of probability. Inferential statistical analysis infers properties of a population, for example by testing hypotheses and deriving estimates.

**2.3 Joint Probability & Marginal Probability**
* **Joint Probability:** is a statistical measure that calculates the likelihood of multiple events occurring together and at the same point in time.

    Suppose there are 2 random variables A and B, $P(A=a,B=b)$ represents the probability of $A = a$ and $B = b$ happening **together**. And the probability of multiple events occurring together at the same time is called **Joint Probability**.
    
* **Marginal Probability:** is the probability of an event irrespective of the outcome of another variable.
    $$
    P(A=a) = \sum_{b} P(A=a, B=b)\\\\
    P(B=b) = \sum_{a} P(A=a, B=b)
    $$
    
**2.4 Conditional Probability**

_Conditional Probability is the probability of one event occurring in the presence of a second event._

**2.5 Relationship between Joint Probability, Marginal Probability & Conditional Probability**

$$
P(A|B) = \frac{P(A,B)}{P(B)}
$$

$$
P(A,B) =P(A)\cdot P(B|A)=P(B)\cdot P(A|B)
$$
    
**2.6 Law of total probability**

If $A_1, A_2, A_3, ..., A_n$ is a finite or countably infinite partition of a sample space (in other words, a set of pairwise **disjoint** events whose **union is the entire sample space** ) and each event $A_i$ is measurable(in other words, $P(A_i)>0$), then for any event $B$ of the same probability space:
$$\begin{align*}
P(B)&=P(B|A_1)P(A_1)+P(B|A_2)P(A_2)+...+P(B|A_n)P(A_n)\\\\
&=\sum_{i=1}^{n}P(B|A_i)P(A_i)
\end{align*}
$$


**2.7 Bayes' Theorem**
    $$
       P(H\ |\ E) =\frac{P(E\ |\ H)\cdot P(H)}{P(E)}
    $$

* $H$ = ***Hypothesis*** whose probability may be affected by data (called evidence below)
* $P(H)$ = ***Priori Probability***, is the estimate of the probability of the hypothesis 
        
* $E$ = ***Evidence*** or Observation, corresponds to new data that were not used in computing the prior probability.

        
* $P(H | E)$ = ***Posterior Probability***, is the probability of H given E, i.e., after E is observed. This is what we want to know: the probability of a hypothesis given the observed evidence.

        
* $P(E\ |\ H)$ = ***Likelihood***, is the probability of observing E given H. As a function of E with H fixed, it indicates the compatibility of the evidence with the given hypothesis. The likelihood function is a function of the evidence, E, while the posterior probability is a function of the hypothesis, H.


* $P(E)$ =  is sometimes termed the ***marginal likelihood*** or "model evidence". This factor is the same for all possible hypotheses being considered (as is evident from the fact that the hypothesis H does not appear anywhere in the symbol, unlike for all the other factors), so this factor does not enter into determining the relative probabilities of different hypotheses.

$$\begin{align*}
Posterior &= \frac{Likelihood * Priori}{Marginal\ Likelihood}\\\\
&= Standard\ Likelihood\ Ratio (LR) * Priori
\end{align*}
$$


* LR > 1, Priori $P(H)$ will be enhanced, $P(E)$ will increase $P(H)$

* LR = 1, Priori $P(H)$ will not change, $P(E)$ will not affect $P(H)$


* LR < 1, Priori $P(H)$ will be weaken, $P(E)$ will decrease $P(H)$

**Based on Bayes' Theorem & Law of total probability:**

$$\begin{align*}
P(H_i|E) &= \frac{P(E| H_i) \cdot P(H_i)}{P(E)}\\\\
&= \frac{P(E| H_i) \cdot P(E)}{\sum_{i=1}^{n}P(E|H_i)\cdot P(H_i)}
\end{align*}
$$

**2.8 Likelihood & Probability**
* **Probability** is about a prediction of possible outcomes, given parameters. 
* **Likelihood** is about an estimation of parameters, given an outcome.

**2.9 Likelihood Function & Probability Function**

For $P(x|\theta)$, there are 2 different perspectives to view it:

* If $\theta$ is known and constant, $x$ is an unknown variable, then $P(x|\theta)$ is a probability function that represents the probability of different $x$.
            
* If random variable $x$ is now known and constant, $\theta$ is an unknown constant, then $P(x|\theta)$ is a likelihood function that represents the probability of $x$ given different $\theta$. Normally, we denote it as $L(\theta|x)$
                
**2.10 Frequentist vs Bayesian**

$x \sim p(x|\theta)$
* **Frequentist:** They think only repeatable random events have probabilities. $\theta$ is an unknown constant, and $x$ is a random variable.
    
    **MLE:** $\theta_{MLE} = \underset{\theta}{argmax}\ \sum_{i=1}^{N}log\ p(x_i|\theta)$

* **Bayesian:** They define probability distributions over possible values of a parameter which can then be used for other purposes. $\Theta$ is a random variable, and has its priori $p(\theta)$ and follows a distribution.
    
    **MAP:** $\theta_{MAP} = \underset{\theta}{argmax}\ p(\theta|x) \propto \underset{\theta}{argmax}\ p(x|\theta) * p(\theta)$
    
    
**2.11 Conjugate**

_If a posterior distribution and prior probability distribution are in the same probability distribution family, then they are called conjugate distributions, and the prior is called a conjugate prior for the likelihood function._

Conjugate priors are useful because they reduce Bayesian updating to modifying the parameters of the prior distribution (so-called hyperparameters) rather than computing integrals.

**Will have a topic focus on this.**


## Example:

We have a coin, and want to estimate the probability of heads $\theta$. In order to estimate it, we flip the coin for 10 times (independent & identically distributed i.i.d), and the results is 6 heads, 4 tails.

## 3. Maximum Likelihood Estimation
Given observations, MLE is to find a parameter that maximize the probability of the observations. 

For a i.i.d sample set, the overall likelihood is the product of likelihood of every sample. And to estimate $\theta$ in the example question, we have likelihood function as
$$
L(\theta|x) = \prod_{i=1}^{N}P(x_i|\theta) = \theta^6(1-\theta)^4
$$

where $\theta$ is an unknown constant, and $x$ is a random variable.  
For mathematical convenient, we convert likelihood function to log likelihood function:

$$
log\ L(\theta|x) = \sum_{i=1}^{N}log(P(x_i|\theta)) = 6log\theta + 4log(1-\theta)
$$

Then calculate the maximum likelihood by derivative:

$$\begin{align*}
& log\ L(\theta|x)' = 0\\\\
\Rightarrow  &\frac{6}{\theta}-\frac{4}{1-\theta} = 0 \\\\
\Rightarrow  &\theta = 0.6
\end{align*}
$$
 
#### MLE of Normal Distribution
Suppose a sample set follows normal distribution i.i.d. $N \sim (\mu, \sigma^2 )$, its likelihood function is
$$
L(\mu,\sigma^2)=\prod_{i=1}^{N}\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}
$$

**Log Likelihood:**
$$
log\ L(\mu,\sigma^2)=-\frac{N}{2}log2\pi-\frac{N}{2}log \sigma^2 -\frac{1}{2\sigma^2}\sum_{i=1}^{N}(x_i-\mu)^2
$$

**Derivatives**
$$
\begin{align*}
\Bigg\lbrace\begin{matrix}\frac{\partial\ log\ L(\mu,\sigma^2)}{\partial\ \mu}=&\frac{1}{\sigma^2}\sum_{i=1}^{N}(x_i-\mu)&=0 
\\\\
\frac{\partial\ log\ L(\mu,\sigma^2)}{\partial\ \sigma^2}=&-\frac{N}{2\sigma^2}-\frac{1}{2\sigma^4}\sum_{i=1}^{N}(x_i-\mu)^2& =0
\end{matrix}
\end{align*}
$$
 
 **Results**
 $$
\begin{align*}
\Bigg\lbrace\begin{matrix}
\hat \mu =& \bar x 
\\\\
\hat \sigma^2 =& \frac{1}{N}\sum_{i=1}^{N}(x_i-\bar x)^2
\end{matrix}
\end{align*}
$$

## 4. Maximum A Posteriori Estimation
MLE treats $\theta$ as an unknown constant, and estimates it by maximizing likelihood function, while MAP thinks $\theta$ as a random variable in some distribution, which is called prior probability distribution. When we estimate $\theta$, we should not only consider likelihood $P(X|\theta)$, but also priori $P(\theta)$. And the best estimate of $\theta$ is the $\theta$ that maximize $P(X|\theta)P(\theta)$. 

Now, we want to maximize $P(X|\theta)P(\theta)$. Since $P(X)$ is a fix value, we can find $\theta$ by maximizing $\frac{P(X|\theta)P(\theta)}{P(X)}$. Based on Bayes Theorem, $P(\theta|X)=\frac{P(X|\theta)P(\theta)}{P(X)}$, and $P(\theta|X)$ is the posterior probability of $\theta$, our target become maximizing a posteriori.

--------------
NOTE: __MAP estimation as regularization of MLE, $P(\theta)$ is the regularizer.__
--------------

***Equation:***
$$
\begin{align*}
\hat\theta_{MAP} &= \underset{\theta}{argmax}\ P(\theta|X) 
\\\\&=\underset{\theta}{argmax}\ \frac{P(X|\theta)P(\theta)}{P(X)}
\\\\&\propto \underset{\theta}{argmax}\ P(X|\theta) P(\theta)
\end{align*}
$$

Back to the example, now we have a priori. We normally assume that the coin is a fair one, and it follows normal distribution. So we assume $\theta = 0.5$ and $\sigma^2=0.1$, and $P(\theta)$ is:
$$
\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}=\frac{1}{\sqrt{0.2\pi}} e^{-\frac{(x_i-0.5)^2}{0.2}}
$$
**Estimate $\hat\theta$:**
$$
\begin{align*}
\hat\theta_{MAP} =\ &\underset{\theta}{argmax}\ P(X|\theta) P(\theta)\\\\
=\  & \underset{\theta}{argmax}\ \theta^6\times(1-\theta)^4\times \frac{1}{\sqrt{0.2\pi}} e^{-\frac{(x_i-0.5)^2}{0.2}}\\\\
\Rightarrow\ &\underset{\theta}{argmax}\ log\ P(X|\theta) P(\theta)
\\\\
 =\ &\underset{\theta}{argmax}\ 6\theta + 4(1-\theta)+ log(\frac{1}{\sqrt{0.2\pi}}) - \frac{(x_i-0.5)^2}{0.2}\\\\
 \end{align*}
$$

After Derivative, $\hat\theta_{MAP} \approx 0.529$

## 5. Bayesian Estimation
Bayesian Estimation is an extension of MAP. Similar to the MAP, Bayesian Estimation assumes that $\theta$ is a random variable, but instead of estimating a specific value for $\theta$, **it estimates the probability distribution of $\theta$. (This is the difference between MAP and Bayesian Estimation)**  In Bayesian Estimation, $P(X)$ ***can not be neglected.***

In our example, we've already known $X$**(events)**, so the probability distribution of $\theta$ given events is $P(\theta|X)$, and this is a posterior distribution. 

If this posterior distribution has a narrow range, then the estimation is more precise, on the contrary, if the posterior distribution has a large range, then the accuracy of estimation is lower.

**Bayes Theorem:**
$$\begin{align*}
P(\theta|X) &= \frac{P(X| \theta) \cdot P(\theta)}{P(X)}
\end{align*}
$$

When X is a continuous random variable, $P(X) = \int_{\theta}P(X|\theta)P(\theta)d\theta$, and

$$\begin{align*}
P(\theta|X) &= \frac{P(X| \theta) \cdot P(\theta)}{\int_{\theta}P(X|\theta)P(\theta)d\theta}
\end{align*}
$$

It is impossible to calculate this integral $P(X) = \int_{\theta}P(X|\theta)P(\theta)d\theta$, the option is we choose using conjugate distribution. 

# ***With a conjugate prior the posterior is of the same type, in our example, for binomial likelihood the beta prior becomes a beta posterior.***

# **To be fixed**

## Conclusion
MLE, MAP and Bayesian Estimation can be viewed as 3 steps of estimation, for each step, we use more information.

MLE and MAP, both of them assume $\theta$ is an unknown constant. The difference here is MAP takes priori into consideration $\Big(P(X|\theta)P(X)\Big)$, while MLE $\Big(P(X|\theta)\Big)$ does not. 

Bayesian Estimation, on the other hand, assumes $\theta$ is an unknown random variable, and gives us the posterior distribution of $\theta$ given $X$, $P(\theta|X)$.


## References
* [Bayesian inference From Wikipedia](https://en.wikipedia.org/wiki/Bayesian_inference#:~:text=Bayesian%20inference%20is%20a%20method,and%20especially%20in%20mathematical%20statistics.)
* [贝叶斯估计、最大似然估计、最大后验概率估计](https://blog.csdn.net/Quincuntial/article/details/80528489)
* [Conjugate priors: Beta and normal](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading15a.pdf)