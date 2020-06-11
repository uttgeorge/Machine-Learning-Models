# Evaluation Metrics - Classification

仅对分类问题做介绍

## Confusion Matrix 混淆矩阵


<!--<tbody>
<td>
<table class="wikitable" style="border:none; float:left; margin-top:0;">
<tbody><tr>
<th style="background:none; border:none;" colspan="2" rowspan="2">
</th>
<th colspan="2" style="background:none;">Actual class
</th></tr>
<tr>
<th>P
</th>
<th>N
</th></tr>
<tr>
<th rowspan="2" style="height:6em;"><div style="display: inline-block; -ms-transform: rotate(-90deg); -webkit-transform: rotate(-90deg); transform: rotate(-90deg);;">Predicted<br>class</div>
</th>
<th>P
</th>
<td><b>TP</b>
</td>
<td>FP
</td></tr>
<tr>
<th>N
</th>
<td>FN
</td>
<td><b>TN</b>
</td></tr>
</tbody></table>
</td>
</tbody>-->

**TP:** 预测为真，实际为真。
**TN:** 预测为假，实际为假。
**FP:** 预测为真，实际为假。Type I Error
**FN:** 预测为假，实际为真。Type II Error



### 1. Accuracy 准确率

$Accuracy=\frac{TP+TN}{TP+FN+TN+FP}=\frac{N_{correct}}{N_{total}}=\frac{正确分类个数}{总样本个数}$

$Error/ Rate=1-Accuracy$

##### 缺点
当样本不均衡(unbalanced)，且我们对minority更感兴趣的时候，accuracy不具有参考性。


```
例子：
当检测癌症的时候，100个人中只有1个人是患者。
假设有一个模型认为这100个人都是健康的，模型的准确率是99%。
这样看上去很好，但很明显，这样的模型是没有意义的。
```


这个称为：accuracy paradox。

### 2. Precision 精确率/查准率

$Precision=\frac{TP}{TP+FP}$

Precision是true positive比上所有 _预测为正_ 的样本个数。
**Precision衡量的是： 在所有预测为正的样本中，实际有多少是正的。是一种衡量模型可信度的metric，或者说对负样本的区分能力。**

比如模型 _预测10个人得了新冠肺炎_ (TP+FP)， _实际上5个人确诊_ (TP)，那么precision就是50%。

### 3. Recall 召回率/查全率

$Recall=\frac{TP}{TP+FN}$

Recall是true positive比上所有 _实际为正_ 的样本个数。
**Recall衡量的是： 在所有实际为正的样本中，预测结果有多少是 _对_ (确实为正) 的。是模型对正样本的识别能力。**

比如 _实际有10个人确诊了新冠肺炎_ (TP+FN)，其中有6个人模型也认为得了肺炎，那么recall就是60%。

### 4. Precision-Recall Tradeoff 

Recall和Precision很难兼得。

通常，当模型尽可能预测出所有的正样本，那么Recall就会比较高，但因为预测更多正样本的同时，也可能有更多的FP（预测为正，实际为负）的样本，Precision就有可能下降。 _这样的模型很aggresive。_
反之，当模型尽可能保证预测出的正样本都是确确实实为正的，那么precision就会提高，但同时，recall可能减少。

### 5. F1-Score

$F1=\frac{2TP}{2TP+FN+FP}=\frac{2\cdot P \cdot R}{P+R}$

F1 是precision和recall的加权调和平均数。F1 score 越高，代表模型越robustness。

当然也可以设置不同的权重：
$$
F_{\beta} = \frac{(1+\beta^2)P\cdot R}{\beta^2 P+ R}
$$

当$\beta=1$时，就是F1分数
当$\beta > 1$时， Precision的权重大，为了提高F分数，recall 越高越好，所以模型看重对正样本的识别能力。
当$\beta<1$时， Recall的权重大，同理，precision越高越好，模型更看重预测的可信度。


### 6. Sensitivity


$Sensitivity=\frac{TP}{TP+FN}$

和Recall一样，实际为正的样本中，预测为正的概率。

### 7. Specificity

$Specificity=1-false\ positive\ rate=\frac{TN}{FP+TN} $

实际为负的样本中，预测为负的概率。

**Specificity和Sensitivity都是基于实际情况下的概率，也就是条件概率。不会因为数据的不平衡而受到影响。**

## 8. ROC Curve

False Positive Rate (1-Specificity) 为x轴。
* 0: 实际为负，预测也都为负。GOOD
* 1: 实际为负，预测都为正。 BAD

True Positive Rate (Sensitivity) 为y轴。
* 0: 实际为正，预测都为负。BAD
* 1: 实际为正，预测都为正。GOOD

![reg](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/Evaluation%20Metrics/media/reg.png)

现在的threshold是0.5，从小到大，从0开始，一旦confusion matrix有改变，就在ROC上对应的点做上标记。另外，右上角顶点(1,0)是所有负样本都预测成正样本的极端情况，原点(0,0)是所有正样本都预测成负样本的极端情况。他们的连线是一条对角线，这条线上：$Sensitivity=1-Specificity$，即正样本中，正确分类的概率等于在负样本，错误分类的概率，这种情况相当于是随机的结果，如果模型结果和随机结果一致，那就相当差了。
左上角顶点是最优情况，右下角为最差。

![ROC](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/Evaluation%20Metrics/media/ROC.png)

AUC(Area Under Curve) 衡量曲线或者折线与x轴所围成的面积，是一个概率值，衡量的是正样本排列在负样本之前的概率值。AUC越大越好。

#### 优点

ROC的优点就是他的缺点， ROC不会收到数据不平衡的影响，能够更加准确的衡量系统的优劣。


#### 缺点

因为ROC不受数据不平衡的影响，所有即便数据数据高度不平衡，ROC也不会怎么改变。所以不能用于数据严重不平衡的模型衡量上。


## 9. PR Curve

PR曲线把Precision作为y轴：
* 0: 所有预测为正的样本实际上都是负样本。
* 1: 所有结果为正样本的预测都正确。

Recall作为x轴：
* 0: 所有实际为正的样本中，没有一个预测对了。
* 1: 预测出了所有正样本。


![pr](https://github.com/uttgeorge/Machine-Learning-Models/blob/master/Evaluation%20Metrics/media/pr.jpg)

同理，改变threshold然后画图。(1,1)顶点是最优情况，代表找到了所有正样本并且全部都正确分类。

#### 优点
$Precision=\frac{TP}{TP+FP}$是指预测为正的样本中，实际为正的比例。所有它有一个优点就是能综合考虑TP和FP，在正样本很少的情况下，能够变化明显。

## PR 和 ROC 的区别
1. 当正负样本不均衡时，ROC变化不大。ROC用于评估分类器整体性能，robustness； PR只关注正样本。
2. ROC过于乐观，如果想评估相同的类别分布下，正样本的预测效果，用PR。
3. 例：信用卡欺诈：使用PR（因为更关注正样本）
    * 要更精准的找到会欺诈的人，提高precision
    * 要更好的预测潜在欺诈的人，提高recall

## CAP Curve

之后写。


## References

* StatQuest with Josh Starmer
* https://www.zhihu.com/question/30643044
* Wikipedia
* https://blog.csdn.net/quiet_girl/article/details/70830796#:~:text=%E3%80%90%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E3%80%91%EF%BC%9A%E3%80%90%E7%BB%93%E6%9E%9C,Re......






