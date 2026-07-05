# Part 3: Advanced Modeling


### 1. Decision Trees
I trained a basic Decision Tree with no limits.
Train accuracy: 0.93
Test accuracy: 0.75
This shows overfitting because train accuracy is way higher than test accuracy. Decision trees have high variance because they greedily fit the training data at every split and basically memorize it.

Then I trained a controlled DT (max depth 5, min samples split 20).
Train accuracy: 0.82
Test accuracy: 0.80
The gap is much smaller now! `max_depth` limits how deep the tree goes, which reduces variance but adds some bias. `min_samples_split` stops it from splitting if there are too few samples, so it doesn't learn noise.


### 2. Gini vs Entropy
Gini test accuracy: 0.807
Entropy test accuracy: 0.794
- Gini formula: 1 - Σ pi²
- Entropy formula: -Σ pi log2(pi)
If Gini = 0, it means the node is perfectly pure and all samples in it belong to the exact same class.


### 3. Random Forest
RF train accuracy: 0.92, test accuracy: 0.77, AUC: 0.83
Top 5 features:
- age: 0.317
- who_man: 0.118
- adult_male: 0.117
- pclass: 0.099
- sex_male: 0.085

Random forest calculates importance by seeing how much a feature reduces Gini impurity on average across all trees. This is different from linear regression coefficients because it doesn't mean a positive or negative linear relationship, it just means the feature is good at splitting the data.

Bagging concept: Bagging uses bootstrap sampling, meaning each tree gets a random sample (with replacement) of the training data. Also, it only looks at a random subset of features (like square root of total features) at each split. Averaging all these different trees together reduces the variance compared to one deep decision tree.


### 4. Gradient Boosting and Ablation
Gradient Boosting AUC: 0.845

Feature ablation study:
- Full RF AUC: 0.8307
- Reduced RF AUC (without 5 worst features): 0.8215

The AUC dropped a little bit. This means the features we removed were actually contributing a small amount and weren't just noise. In production, we might want to deploy the simpler model anyway because it has lower inference cost and is easier to maintain, as long as this small drop in AUC is acceptable.


### 5. Cross Validation
- Logistic Regression CV Mean: 0.8534, Std: 0.0508
- Controlled DT CV Mean: 0.8346, Std: 0.0402
- Random Forest CV Mean: 0.8146, Std: 0.0442
- Gradient Boosting CV Mean: 0.8437, Std: 0.0412

Cross validation gives a more reliable estimate than a single train-test split because it evaluates on multiple different slices of the data, so we don't just get lucky or unlucky with one random split.


### 6. Hyperparameter Tuning
I made a Pipeline with SimpleImputer, StandardScaler, and RandomForest.
Best parameters were: max_depth=5, min_samples_leaf=1, n_estimators=200.
I evaluated 3 x 3 x 2 = 18 configurations. Over 5 folds, that is 90 fits in total.
Grid search checks every single combination which takes a lot of time. Randomized search just picks random combinations, which is faster but might miss the absolute best params.


### 7. Learning Curve
| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 0.9773 | 0.8381 |
| 0.4 | 0.9494 | 0.8517 |
| 0.6 | 0.9273 | 0.8462 |
| 0.8 | 0.9193 | 0.8576 |
| 1.0 | 0.9036 | 0.8577 |

Training AUC decreases as data grows because the high-variance model can't just memorize everything anymore.
Test AUC increases as we give it more data.
Conclusion: The test AUC plateaued at 0.857 between 80% and 100%. This means the model is limited by its capacity now, getting more data won't help much unless we change the model.


### 8. Summary Comparison
| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression (Part 2) | 0.8534 | 0.0508 | 0.8331 |
| Controlled Decision Tree | 0.8346 | 0.0402 | ~0.8000 |
| Random Forest | 0.8146 | 0.0442 | 0.8307 |
| Gradient Boosting | 0.8437 | 0.0412 | 0.8458 |
| Tuned RF Pipeline | 0.8525 | -- | 0.8577 |

Recommendation: I would recommend the Tuned RF Pipeline model. It has the highest Test AUC on the unseen data (0.8577) and its cross-validation score is very strong and stable. Logistic Regression is also good and simpler, but the tuned Random Forest ultimately generalized slightly better.
