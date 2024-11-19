import os
import datetime
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from src.util import time_gaps, normalize_data, read_weights, get_sequences, get_fold_ranges

class StrawberryDataset(Dataset):
    
    def __init__(self, path_to_counts, path_to_weights, k_fold, n_seq, seq_l, n_folds, use_weights=True):
        
        self.mode = 'train'
        
        # intialize the labels and feature dimensions
        self.labels = ['flower','green', 'sw', 'lw', 'pink', 'red', 'gaps']
        self.n_features = len(self.labels)
        self.n_seq = n_seq
        self.seq_l = seq_l
        self.n_folds = n_folds
        self.samples_dim = [self.seq_l, self.n_seq, self.n_features]
        self.use_weights = use_weights
        self.k_fold = k_fold
        
        # get counts
        self.path_to_counts = path_to_counts
        self.months, self.days, self.years = self.get_dates(path_to_counts)
        
        self.X, self.y = self.organize_data()
        self.nX, self.ny = normalize_data(self.X, self.y, len(self.labels))
        
        # get weights
        self.W = read_weights(path_to_weights)
        
        # finalize training data
        self.fnX, self.fny, self.fnX_test, self.fny_test = self.partition_dataset()
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.fnX)
        elif self.mode == 'test':
            return len(self.fnX_test)
        else:
            raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            X = self.fnX[idx].astype(np.float32)
            y = self.fny[idx].astype(np.float32)
        elif self.mode == 'test':
            X = self.fnX_test[idx].astype(np.float32)
            y = self.fny_test[idx].astype(np.float32)
        else:
            raise ValueError("Invalid mode. Choose from 'train' or 'test'.")
    
        return torch.tensor(X), torch.tensor(y)
    
    @staticmethod
    def get_dates(path_to_counts):
        """Gets all the dates of each csv file in the folder"""
        
        dates = []
        for filename in os.listdir(path_to_counts):
            if filename.endswith('.csv'):
                full_date = filename.split('.')[0]
                date_obj = datetime.datetime.strptime(full_date, '%m-%d-%Y')  # Corrected line
                dates.append(date_obj)
            
        dates = list(set(dates))
        dates.sort()
        months = [str(date.month).zfill(2) for date in dates]
        days = [str(date.day).zfill(2) for date in dates]
        years = [str(date.year) for date in dates]
        
        return months, days, years
    
    def organize_data(self, suffx='.csv'):
        """Organizes the data into X and y
        
        Returns:
            X: list of numpy arrays
            y: list of numpy arrays
        """
        
        df_list = []
        for i in range(len(self.months)):
            join_dates = '-'.join((self.months[i], self.days[i], self.years[i]))
            df_list.append(pd.read_csv(self.path_to_counts + join_dates + suffx))
        
        # get the time intervals between each date
        self.delta_t = time_gaps(self.months, self.days, self.years[0])
        
        # initialize data features
        n_features = len(self.labels)
        num_rows = len(df_list[0])
        num_cols = len(df_list)
        
        y_data = np.zeros((num_rows, num_cols))
        X_data = np.zeros((num_rows, num_cols*n_features))
        
        # fill in the data
        col_idx = 0
        for idx, df in enumerate(df_list):
            for label in self.labels:
                
                if label == 'red':
                    red_counts = np.array(df['red'])
                    y_data[:, idx] = red_counts
                    if idx < (num_cols - 1):
                        X_data[:, col_idx] = red_counts
                        col_idx += 1
                elif label == 'gaps':
                    if idx < (num_cols - 1):
                        gaps = self.delta_t[idx]
                        gaps_arr = np.full((num_rows,), gaps)
                        X_data[:, col_idx] = gaps_arr
                        col_idx += 1
                else:
                    class_counts = np.array(df[label])
                    X_data[:, col_idx] = class_counts
                    col_idx += 1
            
        y_data = y_data[:, 1:] # first date not used
        X_data = X_data[:, :-n_features] # last date not used
            
        return X_data, y_data
    
    def partition_dataset(self):
        num_rows = self.ny.shape[0]
        num_cols = self.ny.shape[1] + 1
        _fold_ranges = get_fold_ranges(self.n_seq, num_cols, self.n_folds, ex_dates=1, const=1)

        X_train, y_train, X_test, y_test = [], [], [], []
        if self.n_folds > 1:
            for row_idx in range(num_rows):
                _train_limits = []
                _test_limits = []
                for d_set in range(1, self.n_folds + 1):
                    if d_set != self.k_fold:
                        _train_limits.append(_fold_ranges[d_set - 1])
                    else:
                        _test_limits.append(_fold_ranges[d_set - 1])

                # Train set
                X, y = get_sequences(
                    self.samples_dim, row_idx, _train_limits[0][0], _train_limits[0][self.n_seq],
                    self.delta_t, self.W, self.nX, self.ny, self.use_weights
                )
                X_train.append(X)
                y_train.append(y)

                # Test set
                X, y = get_sequences(
                    self.samples_dim, row_idx, _test_limits[0][0], _test_limits[0][self.n_seq],
                    self.delta_t, self.W, self.nX, self.ny, self.use_weights
                )
                X_test.append(X)
                y_test.append(y)
        else:
            num_dates = int(self.nX.shape[1] / self.samples_dim[2])
            fold_start = 0
            fold_end = num_dates - self.samples_dim[0] + 1
            for row_idx in range(num_rows):
                X, y = get_sequences(
                    self.samples_dim, row_idx, fold_start, fold_end,
                    self.delta_t, self.W, self.nX, self.ny, self.use_weights
                )
                X_train.append(X)
                y_train.append(y)

        X_train_extended = np.concatenate(X_train, axis=0)
        y_train_extended = np.concatenate(y_train, axis=0)
        X_test_extended = np.concatenate(X_test, axis=0)
        y_test_extended = np.concatenate(y_test, axis=0)

        return X_train_extended, y_train_extended, X_test_extended, y_test_extended
