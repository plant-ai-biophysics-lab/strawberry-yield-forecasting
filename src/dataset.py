import os
import datetime
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from src.util import time_gaps, normalize_data

class StrawberryDataset(Dataset):
    
    def __init__(self, path_to_counts):
        
        self.labels = ['flower','green', 'sw', 'lw', 'pink', 'red', 'gaps']
        
        self.path_to_counts = path_to_counts
        self.months, self.days, self.years = self.get_dates(path_to_counts)
        
        self.X, self.y = self.organize_data()
        self.nX, self.ny = normalize_data(self.X, self.y, len(self.labels))
        
    def __len__(self):
        
        # get number of csv files in folder
        return len(os.listdir(self.path_to_counts))
    
    def __getitem__(self, idx):
        
        return self.X[idx], self.y[idx]
    
    @staticmethod
    def get_dates(path_to_counts):
        """Gets all the dates of each csv file in the folder"""
        
        dates = []
        for filename in os.listdir(path_to_counts):
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
        delta_t = time_gaps(self.months, self.days, self.years[0])
        
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
                        gaps = delta_t[idx]
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