import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def prepare_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Ensure there are duplicates
    df = pd.concat([df, df.iloc[:15]], ignore_index=True)
    
    # Introduce a wrong data type by making 'fare' an object (string) with some errors
    df['fare'] = df['fare'].astype(str)
    
    # Ensure missing values
    # deck has > 20% (around 77%)
    # age has < 20% (around 19%)
    
    df.to_csv("titanic.csv", index=False)
    print("titanic.csv created successfully with shape:", df.shape)

if __name__ == "__main__":
    prepare_data()
