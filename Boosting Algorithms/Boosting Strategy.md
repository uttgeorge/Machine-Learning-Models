# Boosting Strategy

In machine learning, boosting is an ensemble meta-algorithm for primarily reducing bias, and also variance in supervised learning, and a family of machine learning algorithms that convert **weak learners** to strong ones. 

Boosting is based on the question posed by Kearns and Valiant (1988, 1989): "Can a set of weak learners create a single strong learner?" 

A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.

(Source:https://en.wikipedia.org/wiki/Boosting_(machine_learning))

## Main Concept

1. Train a base/weak learner from initial training dataset.
2. Adjust the sample distribution based on the performance of base/weak learner.
3. Focus more on the samples that are misclassified.
4. Base on adjusted sample distribution, training a new base/weak learner