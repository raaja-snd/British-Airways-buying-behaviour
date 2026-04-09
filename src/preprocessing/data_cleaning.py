

def drop_outliers(df,labels):
    '''
    Function that drops outliers from the British Airways dataset, 
    when the length of stay is more than 365. Flight books are yearly 
    and roundtrips cannot exceed a year

    Args:
        df(pd.Dataframe): The british Airways dataset
        labels(pd.Series): The labels
    '''
    stay_greater_than_year = df.query("length_of_stay >365").index
    df = df.drop(index=stay_greater_than_year)
    labels =labels.drop(index=stay_greater_than_year)
    return df, labels