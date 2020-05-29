# KNN

-------

## Overview (近朱者赤，近墨者黑。)
K-nearest Neighbor algorithm (k-NN) is a non-parametric method used for classification and regression.

The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other. 

It only based one or several nearest neighbors to determine its category or value. And since it only rely on limited number of neighbors, KNN is more preferable when the data overlap a lot.
![](media/15907186358436/15907194200984.jpg)

-------

## Basic Concept
We measure the distance from **query point** to each sample points, find the **K-nearest neighbors**, and determine the **label** or **value** of this query point.
### 1. Distance Metrics
We use **distance** to measure the similarity of different samples. Euclidean distance, Minkowski Distance, Mahalanobis Distance, Haversine Distance and Manhattan Distance are most commonly used.

* Euclidean distance
    * Euclidean distance
$$
\begin{align*}
& Distance\ from\ x_1\ to\ x_2:\\\\
& d(x_1,x_2) = \sqrt{\sum_{i=1}^{N}(x_{1i}-x_{2i})^2}
\end{align*}
$$
    But normally scales in different features are different, or in other words different scale in different dimension, we prefer to standardize the distance.
    
    * Standardized Euclidean Distance
$$
\begin{align*}
& Distance\ from\ x_1\ to\ x_2:\\\\
& x_{I}^{'} = \frac{x_i-\mu_i}{s_i}\\\\
& \mu: mean,\ s:standard\ variance\\\\
& d(x_1,x_2) = \sqrt{\sum_{i=1}^{N}(\frac{x_{1i}-x_{2i}}{s_i})^2}
\end{align*}
$$    
    
* Manhattan Distance

    $$
    \begin{align*}
    d(x_1,x_2) = \sum ( \left |x_{1i} - x_{2i} \right |)
    \end{align*}
    $$
    
* Minkowski Distance
    
    $$
    \begin{align*}
    & d(x_1,x_2) = \Big(\sum_{i=1}^{N} \left | x_{1i} - x_{2i}\right |^p\Big )^{(1/p)}
    \end{align*}
    $$
    
    * When p = 1, it is Manhattan Distance;
    * When p = 2, it is Euclidean Distance;
    * When p is infinite, it is Chebyshev Distance.
    
    **Pitfalls:**
    1. Scale matters, if scales in different dimensions are different, these distance can not be applied unless standardizing them.
    2. Distribution in each dimension might be different.
    3. We assume that features are independent to each other.
    
* Mahalanobis Distance

    Since the disadvantage of Euclidean distance, we introduce Mahalanobis Distance. This distance utilizes covariance matrix $\Sigma$ to offset the impact of different scales. 
    $$
    \begin{align*}
    & d(X,Y) = \sqrt{(X-Y)^T\Sigma^{-1}(X-Y)}
    \end{align*}
    $$   
    
* Haversine Distance

### 2. Choosing K
#### a. Impact of K

* Large K
If K is too large, then we will take those not similar points into consideration. The system bias will be low but the variance will be high. The system will be very robust.
* Small K
If K is too small, the result may easily be affected by noise. The bias will be high and variance will be low. The system will be very sensitive. 

#### b. How to choose the best K

* **Using Cross Validation**

* Empirical rule: K is normally less than the square root of number of sample size.

### 3. Decision Rule

* **Classification:**  The majority wins.
* **Regression:** Mean of K-nearest neighbors.


-------

## Algorithms


