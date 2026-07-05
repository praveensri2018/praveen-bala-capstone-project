# Part 1: Capstone Data Cleaning


I am using the Titanic dataset for this part. It has many rows and columns like age and fare, and some missing values and duplicates which is good for the assignment.


To run this:
first do `python prepare_dataset.py` to get the data
then do `python maincode.py` to clean it and make the plots.


### 1. Missing Values
The deck column has a lot of missing values (like over 70%). So I didn't fill it.
For age, I filled it with the median. Median is better than mean because if there are outliers, the mean gets pulled too much but median stays in the middle.


### 2. Duplicates
I found 125 duplicate rows and removed them. This changed the null percentage slightly because duplicates had different nulls.


### 3. Data types
I changed fare to numeric because it was object, and changed sex to category. Memory usage went down after this.


### 4. Skewness
The fare column has the highest skewness (4.56). This is positive skew, meaning there are a few really high fares pulling the tail to the right. If we use mean for missing fares, it would be too high because of those few rich people.


### 5. Outliers
Using IQR method, I found 39 outliers in age and 102 in fare. I am keeping them for part 2 because they are real data, rich people and old people exist and they are important for the model.


### 6. Plots
- **line_plot.png**: This is just age plotted. It goes up and down.
- **bar_chart.png**: Shows fare by class. First class is much higher than others.
- **histogram.png**: Histogram of fare. It is skewed right, most people paid very little.
- **scatter_plot.png**: Age vs fare. It shows there is no strong relationship between them.
- **box_plot.png**: Age by survived. It looks almost same but younger people survived a bit more.
- **correlation_heatmap.png**: Pclass and fare have the highest correlation (-0.55). This makes sense because lower class number means better ticket so higher fare. It is not exactly causal, maybe wealth is the third variable causing both.


### 7. Imputation strategy
For the top 2 skewed columns (fare and sibsp), I chose median. Mean for fare is 32 but median is 14. Mean is too high because of skew, so median is better.


### 8. Spearman vs Pearson
I checked spearman and pearson correlations. The biggest difference was for fare and sibsp. Spearman was higher (0.41 vs 0.13). This means they increase together but not in a straight line. I will use Spearman for feature selection in Part 2 because it is better for skewed data.


### 9. Grouped aggregation
I grouped fare by class. First class has the highest mean and highest standard deviation. High standard deviation is bad for model because just knowing they are first class doesn't tell you exactly what they paid, there is too much spread.
The ratio of highest mean to lowest mean is 6.22, which is big enough to be a good feature.
