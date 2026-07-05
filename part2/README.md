# Part 2: Supervised Machine Learning


This is my code for regression and classification.


### 1. Data Setup
I loaded `cleaned_data.csv`. My regression label is `fare` and classification label is `survived`.
I put everything else in X, but I dropped `fare`, `survived`, and `alive` from X so the model doesn't cheat (data leakage).


### 2. Encoding
For `class` (First, Second, Third), I used label encoding (0, 1, 2) because there is a natural order. First class is better than Third class.
For others like `sex` and `embarked`, I used one-hot encoding. This drops one column to avoid multicollinearity. We use one-hot because they don't have an order. If we used label encoding for sex (e.g. male=1, female=2), the model would think female is "greater" than male which is a false ordinal relationship.


### 3. Scaling
I used train_test_split. Then I fit StandardScaler ONLY on the training data. If I fit it on the whole dataset, the test set's mean and standard deviation would leak into the training process, and the model would have seen the test data before predicting.


### 4. Regression (Predicting Fare)
Linear Regression MSE was 731.14 and R2 was 0.46. 

Top 3 coefficients:
- pclass: -14.71
- class: 14.71
- who_woman: 11.35

A large positive coefficient (like class) means when this feature increases, the predicted fare goes up by that many units. A large negative coefficient means the fare goes down. 

Ridge Regression comparison:
| Model | MSE | R2 |
|---|---|---|
| Linear | 731.14 | 0.46 |
| Ridge | 729.54 | 0.46 |

Ridge got a slightly better MSE. Ridge is different because the `alpha` parameter adds a penalty that shrinks the coefficients. So Ridge doesn't rely too much on just one feature, unlike normal Linear regression.


### 5. Classification (Predicting Survived)
I checked class imbalance. `survived=1` is around 41%, which is > 35%. But I used `class_weight='balanced'` in LogisticRegression just to be safe and give both classes equal importance.

My AUC is 0.8331. AUC means how well the model separates survivors from non-survivors. 0.83 is pretty good.

Formulas:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

For predicting Titanic survival, maybe Recall is more important because we want to find as many survivors as possible. False negatives (missing a survivor) is bad. So we want high recall.

**Threshold Sensitivity Table:**
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.6709 | 0.8030 | 0.7310 |
| 0.40 | 0.7353 | 0.7576 | 0.7463 |
| 0.50 | 0.7667 | 0.6970 | 0.7302 |
| 0.60 | 0.8269 | 0.6515 | 0.7288 |
| 0.70 | 0.8537 | 0.5303 | 0.6542 |

The threshold that maximizes F1 is 0.40 (F1=0.7463).
If we want to optimize for recall, we should LOWER the threshold (e.g. to 0.30). The cost of doing this is that precision drops, meaning we get more false positives.


### 6. Regularization Experiment
I trained another Logistic Regression with C=0.01. C controls the inverse of regularization strength. Small C means strong penalty.

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (Baseline) | 0.7667 | 0.6970 | 0.8331 |
| C=0.01 (Strong) | 0.7541 | 0.6970 | 0.8319 |

Reducing C actually worsened the performance slightly because the model was penalized too much.


### 7. Bootstrap Confidence Interval
I ran 500 bootstrap samples to compare AUC of C=1.0 and C=0.01.
- Mean AUC Difference: 0.0014
- 2.5th percentile: -0.0125
- 97.5th percentile: 0.0154

Since the 95% confidence interval goes from negative to positive (it includes zero), the difference between the two models is not reliable. They are basically the same on this dataset.
