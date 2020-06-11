# Evaluation Metrics - Regression

Only regression evaluation metrics.

* $y$: Actual y
* $\hat{y}$: Predicted y
* $N$: Sample size

## 1. MAE: mean absolute error

MAE is l1-norm loss:

$$MAE(y,\hat{y})=\frac{\sum_{i=1}^{N}|y-\hat{y}|}{N}=\frac{\sum_{i=1}^{N}|e|}{N}$$

## 2. MSE: mean square error

MSE is l2-norm loss:

$$
MSE(y,\hat{y})=\frac{\sum_{i=1}^{N}(y-\hat{y})^2}{N}=\frac{\sum_{i=1}^{N}e^2}{N}
$$

## 3. RMSE: root mean square error

RMSE is the root of MSE:

$$
RMSE = \sqrt{MSE}=\sqrt{\frac{\sum_{i=1}^{N}(y-\hat{y})^2}{N}}
$$

This is the most common used metric. But it is sensitive to outliers because it is using average error. 

**RMSE is not robust.**

## 4. MAPE: mean absolute percentage error

$$
MAPE=\frac{1}{N}\sum_{i=1}^{N} \left | \frac{y-\hat{y}}{y} \right |
$$

Solve the impact of outliers.

## 5. $R^2$: Coefficient of determination

$$
R^2=1-\frac{SSE}{SST}=\frac{SSR}{SST}=1-\frac{\sum_{i=1}^{N}(y-\hat{y})^2}{\sum_{i=1}^{N}(y-\bar{y})^2}
$$

The percentage of variance that can be explained by regression model.
But when there are more variables, r-square would definitely increase. So we have to use adjusted one.


## 6. $Adjusted\ R^2$: Adjusted coefficient of determination

$$
R_{Adjusted}^2=1-(1-R^2)\frac{N-1}{N-p-1}
$$

where $p$ is the total number of explanatory variables in the model.


## References

* Wikipedia


