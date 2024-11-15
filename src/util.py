import numpy as np

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def time_gaps(months, days, year): 
    """ This functions generates the time gaps vector
    Args:
        months (string list): list with strings of months
        days (string list): list with strings of days
        year (string): string of year variable
    Returns:
        string list: list containing time gaps
    """
    
    year = int(year)
    dates = [datetime(year, int(month), int(day)) for month, day in zip(months, days)]
    differences = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    
    return differences

def normalize_data(X, y, n_features):
    """ This function normalizes the input data. Separately normalizes the features data from the stages' counts and the time intervals.
    Args:
        X (np.array): input feature data 
        y (np.array): input targe data
        n_features (int): number of features in the input data
    Returns:
        np.array, np.array: normalized x and y data as arrays
    """
    
    saved_gaps = []
    for i in range(int(len(X[0])/n_features)): # add variable for number of features
        saved_gaps.append(X[0][(i*n_features)+(n_features-1)]) # same here

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    nX_Data = X_scaler.fit_transform(X)

    gaps_data = np.array(saved_gaps).reshape(-1, 1)
    gaps_scaler = MinMaxScaler(feature_range=(0, 1))
    n_gaps_Data = gaps_scaler.fit_transform(gaps_data)

    for i in range(int(len(X[0])/n_features)):
        nX_Data[0][(i*n_features)+(n_features-1)] = n_gaps_Data[i][0]
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    ny_Data = y_scaler.fit_transform(y)
    
    return nX_Data, ny_Data