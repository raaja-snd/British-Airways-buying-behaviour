from sklearn.model_selection import train_test_split
import pandas as pd


def get_train_test_data(df:pd.DataFrame,labels:pd.Series):
    return train_test_split(df,labels,test_size=0.2,random_state=42,stratify=labels) 