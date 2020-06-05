# Ensemble Method - Bagging

## Bagging

**Bootstrap aggregating**, also called **bagging** (from bootstrap aggregating), is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach.

## Algorithm

1. **Bootstrapping:** Sampling with replacement to get a bootstrap dataset with n samples.
2. **Model:** Build a classifier or regressor based on these n samples.
3. **Repeat** step 1 & 2 m times (m is an odd number).
4. **Result**:
    * **Classifier**: Vote
    * **Regressor**: Average
    
## Advantages
1. Highly efficient, support parallel computing, the time complexity of it is same order of magnitude as its base learner.
2. Bootstrap sampling would leave around 1/3 of data being unselected. When there are a lot of noises in the dataset, bootstrap could reduce the impact of noise, reduce variance and avoid overfitting. Those unselected data is called "Out of Bag Data", and can be used to evaluate/validate. 

## References
* Wikipedia