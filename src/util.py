import csv
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

def read_weights(path):
    """ This function reads the phenological weights from a csv file and return them into a list variable
    Args:
        path (string): path to phenological weights .csv file
    Returns:
        list: list with weights for phenological classes
    """
    W_matrix = []
    with open(path,newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            W_matrix.append(row)    
    return W_matrix

def get_fold_ranges(n_seq,n_dates,n_folds,ex_dates,const):
    """ This function computes the start and stop sample idx for each of the folds given a time-series dataset structure
    Args:
        n_seq (int): number of sequences desired for model input
        n_dates (int): number of samples in the time series dataset
        n_folds (int): number of folds the dataset is divided into
        ex_dates (int): number of dates that will be excluded from the dataset so folds are even. This is computed by the user
        const (int): constant to make folds even. Not a formal justification why this is needed.
    Returns:
        int list: list containing start stop sample idx for train/test for the n_folds
    """
    _fold_ranges = []
    for k_fold in range(n_folds): #try stacking this for into for below
        _fold_range = []
        for i in range(k_fold*(int((n_dates-ex_dates)/n_folds)), (const+k_fold*(int((n_dates-ex_dates)/n_folds))+n_seq)):
            _fold_range.append(i)
        _fold_ranges.append(_fold_range)
    return _fold_ranges

def get_sequences(samples_dim,row_idx,start,stop,time_spacing,W_matrix,X_Data,y_Data,phenological=True):
    """ This function generates the X and y data to be used as input wether it is used for training or testing. 
    If not specified, the phenological weights are applied by default.
    Args:
        samples_dim (list): data dimensions
        row_idx (int): index for the rows in X_Data
        start (int): start value used in for loop for extraction
        stop (int): stop value used in for loop for extraction
        time_spacing (int list): list containing consecutive intervals between dates
        W_matrix (list): list containing weights for all phenological classes at different days from flower to ripe fruit
        X_Data (np.array): matrix containing X features data
        y_Data (np.array): matrix containing y target data
        phenological (bool, optional): by default this is set to True.
    Returns:
        float list, float list: X and y data
    """
    X, y = [], []
    for i in range(start,stop):
        sum_gaps = []
        gap_sum = 0
        gaps_vector = time_spacing[i:i+samples_dim[0]] # This computes te cumulative days between dates in the sequence
        max_gaps = np.max(time_spacing)
        for gap_idx in range(len(gaps_vector) - 1,-1,-1):
            gap_sum += gaps_vector[gap_idx]
            sum_gaps.append(gap_sum)
        sum_gaps.reverse()
        max_sum_gaps = np.max(sum_gaps)
        raw_input_seq = []
        input_sequence = []
        input_target = []
        _weights = []
        occ_fy = 1+((i+samples_dim[0]-1)/len(time_spacing))
        for date_idx,date in enumerate(sum_gaps):
            raw_input_seq.append(X_Data[row_idx,(date_idx+i)*(samples_dim[2]):((date_idx+i)*(samples_dim[2]))+(samples_dim[2])])
            if phenological:
                float_weights = [float(element) for element in W_matrix[date]]
                float_weights.append(1.0)
                _weights.append(float_weights)
                occ_fx = 1+((i+date_idx)/len(time_spacing))
                #corrected_input = np.multiply(raw_input_seq, [0.7, 0.7, 0.8, 0.8, 0.8, 1.1, 1.0])#[1.3, 1.2, 1.3, 1.17, 1.5, 1.0, 1.0]
                #corrected_input = np.multiply(raw_input_seq, [occ_fx, occ_fx, occ_fx, 1.0, 1.0, 1.0, 1.0]) # Using prob_occ
                #input_sequence = np.multiply(_weights, corrected_input)
                input_sequence = np.multiply(_weights, raw_input_seq)
            else:
                input_sequence = np.array(raw_input_seq)
        input_target.append(y_Data[row_idx,i+samples_dim[0]-1]*1.0)
        #norm_gaps_vector = [x / max_gaps for x in gaps_vector] # This normalize the gaps vector
        norm_sum_gaps = [x / max_sum_gaps for x in sum_gaps] # This normalize the cumulative gaps vector 
        input_sequence[:,6] = norm_sum_gaps#norm_gaps_vector # use gaps vector
        input_X = input_sequence[:,:] # Remove certain count features
        #print('input seq.',input_sequence)
        #input_X = input_sequence
        X.append(input_X)
        y.append(input_target)
    return X,y