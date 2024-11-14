import pandas as pd
import numpy as np
from datetime import datetime
import csv
import copy
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def extract_dates_from_filenames(folder_path,year):
    """Extracts the dates found in a folder
    Args:
        folder_path (string): path to folder
        year (int): year found in the strings
    Returns:
        list, list, string: lists for months and days found in folder. Also year  as single string
    """
    dates = []
    for filename in os.listdir(folder_path):
        if year==2023:
            try:
                date_str = filename.split('_')[0]
                date_obj = datetime.strptime(date_str, '%m-%d-%Y')
                dates.append(date_obj)
            except (IndexError, ValueError):
                continue
        elif year==2022:
            try:
                date_str = filename.split('_')[0]
                date_obj = datetime.strptime(date_str, '%m-%d-%Y')
                dates.append(date_obj)
            except (IndexError, ValueError):
                continue
    # Remove duplicates by converting to a set and back to a list
    unique_dates = list(set(dates))
    # Sort the dates in chronological order
    unique_dates.sort()
    months = [str(date.month).zfill(2) for date in unique_dates]
    days = [str(date.day).zfill(2) for date in unique_dates]
    return months, days, str(year)

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

def read_weights(path):
    """ This function reads the phenological weights from a csv file and return them into a list variable
    Args:
        path (string): path to phenological weights .csv file
    Returns:
        list: list with weights for phenological classes
    """
    if path is None:
        path = '/Users/andres/Documents/strawberry-forecasting/data/weights/weights.csv'
    W_matrix = []
    with open(path,newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            W_matrix.append(row)    
    return W_matrix

def time_gaps(months,days,year): 
    """ This functions generates the time gaps vector
    Args:
        months (string list): list with strings of months
        days (string list): list with strings of days
        year (string): string of year variable
    Returns:
        string list: list containing time gaps
    """
    year = int(year)
    # Convert strings to datetime objects
    dates = [datetime(year, int(month), int(day)) for month, day in zip(months, days)]
    # Calculate the difference in days between consecutive dates
    differences = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    return differences

def data_loader(path,months,days,year,_str_features,prefix,suffix,red_feature=True):
    """ This function reads yield data from the detection results files for every date and arranges data into X and y
    Args:
        path (string): string containing path to folder where detection results are located
        months (string list): list with strings of the months in the dates of every file for detection results
        days (string list): list with strings of the days in the dates of every file for detection results
        year (string): string of year in the dates of every file for detection results
        _str_features (string list): list containing the features names as strings
        prefix (string): string used to find the right file in the path
        suffix (string): string used to find the right file in the path
        red_feature (bool, optional): Not justified why this is needed. Check this.
    Returns:
        np.array, np.array: X and y data 
    """
    df_list = []
    for i in range(len(months)):
        join_date = '-'.join([months[i], days[i], year])
        if prefix == '':
            df_list.append(pd.read_csv(path + join_date + suffix))
        else:
            df_list.append(pd.read_csv(path + prefix + join_date + suffix))

    time_spacing = time_gaps(months, days, year) # Get time gaps (in days) between dates
    if red_feature:
        n_features = len(_str_features)
    else:
        n_features = len(_str_features) - 1 # red class is removed from features as this is the target variable
    num_rows = len(df_list[0])
    num_cols = len(df_list)
    y_data = np.zeros((len(df_list[0]), num_cols)) # Array to contain y data
    X_data = np.zeros((len(df_list[0]), n_features * num_cols)) # Array to contain X data
    X_col_idx = 0 # Column idx for adding data in X data 

    for df_idx,df in enumerate(df_list):
        try:
            for i_class in _str_features:
                if i_class == 'red':
                    red_count = np.array(df['red'])
                    y_data[:,df_idx] = red_count
                    if red_feature:
                        if df_idx < (num_cols-1):
                            X_data[:,X_col_idx] = red_count
                            X_col_idx += 1
                elif i_class == 'gaps':
                    if df_idx < (num_cols-1):
                        gaps = time_spacing[df_idx]
                        gaps_array = np.full((num_rows,), gaps)
                        X_data[:,X_col_idx] = gaps_array # remove last column
                        X_col_idx += 1
                else:
                    class_count = np.array(df[i_class])
                    X_data[:,X_col_idx] = class_count
                    X_col_idx += 1
            y_Data = y_data[:, 1:] # Remove first column as red count in the first date is not used
            X_Data = X_data[:, :-n_features] # Remove last 6 columns as X features for last date are not used
        except KeyError:
            print(f"DataFrame: {df_list.index(df) + 1}, Column or Row not found.")
    return X_Data, y_Data

def normalize_data(x_data,y_data,num_features,red=False):
    """ This function normalizes the input data. Separately normalizes the features data from the stages' counts and the time intervals.
    Args:
        x_data (np.array): input feature data 
        y_data (np.array): input targe data
        num_features (int): number of features in the input data
        red (bool, optional): Check why this variable is defined. Defaults to False.
    Returns:
        np.array, np.array: normalized x and y data as arrays
    """
    if red:
        saved_red = []
        for row in x_data:
            saved_red_row = []
            for date in range(int(len(row)/num_features)): # add variable for number of features
                saved_red_row.append(row[(date*num_features)+(num_features-2)]) # same here
            saved_red.append(saved_red_row)
        red_scaler = MinMaxScaler(feature_range=(0, 1))
        n_red_Data = red_scaler.fit_transform(saved_red)
        for i in range(len(x_data)):
            for j in range(int(len(row)/num_features)): # add variable for number of features
                x_data[i][(j*num_features)+(num_features-2)] = n_red_Data[i][j] # same here
        nX_Data = x_data
    else:
        saved_gaps = []
        for i in range(int(len(x_data[0])/num_features)): # add variable for number of features
            saved_gaps.append(x_data[0][(i*num_features)+(num_features-1)]) # same here

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        nX_Data = X_scaler.fit_transform(x_data)

        gaps_data = np.array(saved_gaps).reshape(-1, 1)
        gaps_scaler = MinMaxScaler(feature_range=(0, 1))
        n_gaps_Data = gaps_scaler.fit_transform(gaps_data)

        for i in range(int(len(x_data[0])/num_features)):
            nX_Data[0][(i*num_features)+(num_features-1)] = n_gaps_Data[i][0]
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    ny_Data = y_scaler.fit_transform(y_data)
    return nX_Data, ny_Data

def get_phenological_yield(X_train_extended, W_matrix, samples_dim):
    """ Computes the phenological-based yield from the input features.
    Args:
        X_train_extended (np.array): Array containing all training data per fold.
        W_matrix (np.array): Matrix containing the phenological weights
        samples_dim (list): data dimensions
    Returns:
        np.array: yield from the weighted input features.
    """
    X_prev_stages = X_train_extended
    W_train = []
    for seq in X_prev_stages:
        delta_t_sum = 0.0
        delta_t_vector = []
        for i in range(samples_dim[0]-1,-1,-1):
            delta_t_sum += seq[i][samples_dim[2]-1]
            delta_t_vector.append(delta_t_sum)
        delta_t_vector.reverse()
        w_cum_yield = 0.0
        for date_i,seq_i in enumerate(seq):
            float_weights = [float(element) for element in W_matrix[int(delta_t_vector[date_i])]]
            w_input = np.sum(np.multiply(seq_i[0:5],float_weights[:-1]))
            w_cum_yield += w_input
        w_yield = w_cum_yield / samples_dim[0]
        W_train.append(w_yield)
    W_train_extended = np.array(W_train)
    return W_train_extended
