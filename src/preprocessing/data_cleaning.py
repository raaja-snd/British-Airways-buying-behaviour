

def drop_outliers(df,labels):
    stay_greater_than_year = df.query("length_of_stay >365").index
    df = df.drop(index=stay_greater_than_year)
    labels =labels.drop(index=stay_greater_than_year)
    return df, labels