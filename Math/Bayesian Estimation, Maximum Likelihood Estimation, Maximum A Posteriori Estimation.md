# Bayesian Estimation, Maximum Likelihood Estimation, Maximum A Posteriori Estimation

## **1. Overview**

Basic concepts of Bayesian Estimation, MLE, MAP

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

 



## References
* [Bayesian inference From Wikipedia](https://en.wikipedia.org/wiki/Bayesian_inference#:~:text=Bayesian%20inference%20is%20a%20method,and%20especially%20in%20mathematical%20statistics.)
* [贝叶斯估计、最大似然估计、最大后验概率估计](https://blog.csdn.net/Quincuntial/article/details/80528489)