import os
import datetime
from torch.utils.data import Dataset

class StrawberryDataset(Dataset):
    
    def __init__(self, path_to_counts):
        
        self.path_to_counts = path_to_counts
        self.dates = self.get_dates(path_to_counts)
        
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
            full_date = filename.split('_')[0]
            date_obj = datetime.datetime.strptime(full_date, '%m-%d-%Y')  # Corrected line
            dates.append(date_obj)
            
        dates = list(set(dates))
        dates.sort()
        months = [str(date.month).zfill(2) for date in dates]
        days = [str(date.day).zfill(2) for date in dates]
        years = [str(date.year) for date in dates]
        
        return months, days, years