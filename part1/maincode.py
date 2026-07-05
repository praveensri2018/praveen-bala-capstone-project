import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


warnings.filterwarnings('ignore')
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("- Task 1 -")
    df = pd.read_csv('titanic.csv')
    df_raw = df.copy() 
    print(df.head())


    print("- Task 2 -")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / df.shape[0]) * 100
    
    gt_20_cols = null_pct[null_pct > 20].index.tolist()
    print("cols > 20% nulls:", gt_20_cols)
    
    
    numeric_cols_raw = df_raw.select_dtypes(include=[np.number]).columns
    skewness_raw = df_raw[numeric_cols_raw].skew().abs().sort_values(ascending=False)
    top_2_skewed = skewness_raw.head(2).index.tolist()
    
    lt_20_cols = null_pct[(null_pct <= 20) & (null_pct > 0)].index.tolist()


    for col in lt_20_cols:
        if col in df.select_dtypes(include=[np.number]).columns and col not in top_2_skewed:
            df[col] = df[col].fillna(df[col].median())


    print("- Task 3 -")
    dup_count = df.duplicated().sum()
    
    null_pct_before = (df.isnull().sum() / df.shape[0]) * 100
    df = df.drop_duplicates()
    null_pct_after = (df.isnull().sum() / df.shape[0]) * 100
    print("removed duplicates", df.shape)


    print("- Task 4 -")
    if 'fare' in df.columns and df['fare'].dtype == 'object':
        df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
    
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype('category')


    print("- Task 5 -")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    skewness = df[numeric_cols].skew()
    most_skewed = skewness.abs().idxmax()
    print("highest skewness:", most_skewed)


    print("- Task 6 -")
    for col in ['age', 'fare']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]


    print("- Task 7 -")
    plt.figure(figsize=(10,5))
    plt.plot(df['age'].dropna().reset_index(drop=True))
    plt.title('Age plot')
    plt.savefig('line_plot.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    df.groupby('class')['fare'].mean().plot.bar()
    plt.title('Fare by Class')
    plt.savefig('bar_chart.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    sns.histplot(df[most_skewed].dropna(), bins=20)
    plt.title('Histogram')
    plt.savefig('histogram.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='age', y='fare', data=df)
    plt.title('Age vs Fare')
    plt.savefig('scatter_plot.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    sns.boxplot(x='survived', y='age', data=df)
    plt.title('Age Boxplot')
    plt.savefig('box_plot.png')
    plt.close()
    
    plt.figure(figsize=(10,8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    corr_unstacked = corr_matrix.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1] 


    print("- Task 8a -")
    top_2 = skewness.abs().nlargest(2).index.tolist()
    for col in top_2:
        mean_val = df_raw[col].mean()
        median_val = df_raw[col].median()
        print(f"{col} mean: {mean_val:.4f}, median: {median_val:.4f}")
        
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(median_val)
    


    print("- Task 8b -")
    spearman_corr = df[numeric_cols].corr(method='spearman')
    
    diff_matrix = (spearman_corr - corr_matrix).abs()
    diff_unstacked = diff_matrix.unstack()
    diff_unstacked = diff_unstacked[diff_unstacked.index.get_level_values(0) != diff_unstacked.index.get_level_values(1)]
    diff_unstacked.index = diff_unstacked.index.map(lambda x: tuple(sorted(x)))
    diff_unstacked = diff_unstacked.drop_duplicates()
    
    top_3_diffs = diff_unstacked.nlargest(3)
    print("Spearman diffs:")
    print(top_3_diffs)


    print("- Task 8c -")
    cat_col = 'class'
    num_col = 'fare'
    agg_result = df.groupby(cat_col, observed=False)[num_col].agg(['mean', 'std', 'count'])
    print(agg_result)
    


    df.to_csv('cleaned_data.csv', index=False)
    print("done saving")

if __name__ == "__main__":
    main()
