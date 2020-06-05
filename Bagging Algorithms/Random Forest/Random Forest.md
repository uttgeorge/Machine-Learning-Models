# Random Forest

## Concept
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.

## Difference between Decision Tree & Random Forest
1. Sample Set (Row): RF uses bootstrapping. 1/3 of data will not be selected.
2. Features (Column): In every individual decision tree, suppose there are $N$ features in the dataset, for each decision node, we _**randomly**_ select $N_{sub}\  (N_{sub}<N)$ features to train the decision tree. This could **avoid** always choosing a single best feature and having _high correlation_ between each decision tree. By doing so, we can **improve the generalization** of model, **increase bias** but **decrease variance**.

    **Note: Practically, we use cross validation to get $N_{sub}$, here are some tricks:**
    * **For Classification Problem: $\sqrt{N}$** 
    * **For Regression Problem: $\frac{N}{3}\ but\ greater\ than\ 5.$**
    
1. Decision Trees in Random Forest are _**fully grown and unpruned.**_



## Parameter M
1. **Correlation of any two decision trees:** higher correlation leads to higher error rate.
2. **Ability of classification for each tree:** the stronger the better.

### ***M:***

**M is the only parameter we should set. It is the number of features that would be randomly selected. Decreasing M could lead to decreasing in both _Correlation_ and _Classification Power_ . To find the best M, we commonly use OOB (Out Of Bag) error to find it.**

## How does Random Forest output importance of each feature?
### Classification Problem: 
#### 1. Gini Index
* **VIM**: Variable Importance Measure 
* **GI**: Gini Index
* **N features:** $X_1, X_2, X_3, ..,X_n$

**A.** Gini Index at Node m:
$$
GI_m=1 - \sum_{k=1}^{|K|}p_{m,k}^2
$$
where **k** means k different classes, **m** means the $m_{th}$ node.
For more info, go to [Decision Tree](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/Decision%20Tree/Decision%20Tree.md).

**B.** VIM at Node m:

$$
VIM_m=GI_m-GI_{left}-GI_{right}
$$
It is the decrease of Gini Index when adding a new feature.

**C.** VIM of feature $X_n$ at the $t_{th}$ tree:
$$
VIM_{t,n} = \sum_{m\in M_{X_n}}VIM_m\\\\
$$
where $M_{X_n}$ is the set of nodes in tree $t$ that has feature $X_n$.

**D.** Overall VIM of feature $X_n$(Normalized)
$$
VIM_n = \frac{\sum_{t=1}^{|T|}VIM_{t,n}}{|T|}
$$
where $|T|$ is the total number of trees in random forest.


#### 2. OOB Error Rate (Unbiased Estimate)

When validating by OOB Data, Suppose there are $O$ samples and $X$ misclassifications, the error rate is:
$$
Error\ Rate=\frac{X}{O}
$$
and this is an unbiased estimation.


**A.** Calculate OOB Error Rate 1 at each tree.

**B.** Add noise to the feature we want to know importance, calculate OOB Error Rate 2.

**C.** Importance of feature n:
$$
VIM_n=\frac{\sum_{t=1}^{|T|}(OOB\ ER1 - OOB\ ER2)}{|T|}
$$


## Pro & Con
### Pro
1. Highly efficient, simple
2. _**RF do not need prune and feature selection for each tree since we randomly choose features and samples. RF works fine even the dimension (# of features) is very high.**_
3. _**After training, we can acquire VIM (Variable Importance Measures)**_
4. Robustness, low variance. **It is an unbiased estimate of _generalization error_ .**

$$generalization\ error = bias^2(x)+var(x)+\varepsilon^2$$

6. Not sensitive to missing data.

### Con
1. If noises are too high, then RF could be overfitting.
2. If a feature has too many partitions, then it could affect the decision making.




    
    