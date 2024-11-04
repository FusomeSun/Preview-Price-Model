import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from features.processing import load_stock_data

class SP500DataManager:
    def __init__(self, data_dir='/home/endy/research/Documnets/intern/Price-Prediction-Model/Sprint 2/data/data'):
        self.data_dir = data_dir
        self.tickers_file = os.path.join(data_dir, 'sp500_tickers.pkl')
        self.data_file = os.path.join(data_dir, 'sp500_data.pkl')
        self.last_update_file = os.path.join(data_dir, 'last_update.txt')
        
        os.makedirs(data_dir, exist_ok=True)
        
    def get_sp500_tickers(self):
        if os.path.exists(self.tickers_file):
            with open(self.tickers_file, 'rb') as f:
                return pickle.load(f)
        else:
            tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
            with open(self.tickers_file, 'wb') as f:
                pickle.dump(tickers, f)
            return tickers
    
    def update_data(self, start_date='2000-01-01'):
        tickers = self.get_sp500_tickers()
        
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                all_data = pickle.load(f)
            with open(self.last_update_file, 'r') as f:
                last_update = datetime.strptime(f.read().strip(), '%Y-%m-%d').date()
            start_date = (last_update + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            all_data = {}
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check if the data was updated today or yesterday
        if (datetime.now().date() - last_update).days <= 1:
            print("Data is up to date. No update needed.")
            return
        
        for ticker in tqdm(tickers, desc="Updating S&P 500 data"):
            try:
                new_data = load_stock_data(ticker, start_date, end_date)
                if not new_data.empty:
                    if ticker not in all_data:
                        all_data[ticker] = new_data
                    else:
                        all_data[ticker] = pd.concat([all_data[ticker], new_data]).drop_duplicates().sort_index()
            except Exception as e:
                print(f"Error updating data for {ticker}: {e}")
        
        with open(self.data_file, 'wb') as f:
            pickle.dump(all_data, f)
        
        with open(self.last_update_file, 'w') as f:
            f.write(end_date)
        
        print(f"Data updated successfully up to {end_date}")
    
    def get_data(self, start_date, end_date):
        with open(self.data_file, 'rb') as f:
            all_data = pickle.load(f)
        
        filtered_data = {}
        for ticker, data in all_data.items():
            if not data.empty:
                # Get the date range available for this stock
                stock_start = data.index.min().strftime('%Y-%m-%d')
                stock_end = data.index.max().strftime('%Y-%m-%d')
                
                # Adjust start_date and end_date if necessary
                adjusted_start = max(start_date, stock_start)
                adjusted_end = min(end_date, stock_end)
                
                # Only include the stock if there's data in the adjusted range
                if adjusted_start <= adjusted_end:
                    try:
                        filtered_data[ticker] = data.loc[adjusted_start:adjusted_end]
                    except KeyError as e:
                        print(f"Error getting data for {ticker}: {e}")
                        continue  # Skip this stock and continue with the next one
        
        if not filtered_data:
            raise ValueError(f"No data available for the specified date range: {start_date} to {end_date}")
        
        return filtered_data
    
    def get_latest_date(self):
        if os.path.exists(self.last_update_file):
            with open(self.last_update_file, 'r') as f:
                return f.read().strip()
        return None

    def get_earliest_date(self):
        with open(self.data_file, 'rb') as f:
            all_data = pickle.load(f)
        
        earliest_dates = [data.index.min() for data in all_data.values() if not data.empty]
        return min(earliest_dates).strftime('%Y-%m-%d') if earliest_dates else None

# Usage example
if __name__ == "__main__":
    data_manager = SP500DataManager()
    data_manager.update_data()
    latest_date = data_manager.get_latest_date()
    earliest_date = data_manager.get_earliest_date()
    print(f"Data available from {earliest_date} to {latest_date}")
    
    # Example of getting data for a specific time range
    start_date = '2018-01-01'
    end_date = '2024-07-31'
    try:
        data = data_manager.get_data(start_date, end_date)
        print(f"Loaded data for {len(data)} stocks from {start_date} to {end_date}")
    except ValueError as e:
        print(f"Error: {e}")