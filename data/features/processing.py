import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def load_data(file_name, subdir='data'):
    """
    Loads and normalizes CSV data from a specified subdirectory relative to the script directory.

    Args:
    file_name (str): The name of the CSV file to load.
    subdir (str): The subdirectory where the file is located relative to the script directory.

    Returns:
    pd.DataFrame: The normalized data as a pandas DataFrame.
    """
    
    # Construct the path to the CSV file relative to the script directory
    file_path = os.path.join('..', '..', '..', subdir, file_name)

    # Load the data
    combined_data = pd.read_csv(file_path, index_col=0, parse_dates=False)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data.values)
    
    return scaled_data,combined_data


# Define function to create sequences
def create_sequences(data, seq_length, out_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - out_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length:i + seq_length + out_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def calculate_kdj(data, n=14, m=3):
    low_min = data['Low'].rolling(window=n).min()
    high_max = data['High'].rolling(window=n).max()
    
    rsv = (data['Close'] - low_min) / (high_max - low_min) * 100
    
    k = rsv.ewm(com=m-1, adjust=False).mean()
    d = k.ewm(com=m-1, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_arbr(data, n=26):
    ar = (data['High'] - data['Open']).rolling(window=n).sum() / (data['Open'] - data['Low']).rolling(window=n).sum() * 100
    br = (data['High'] - data['Close'].shift(1)).rolling(window=n).sum() / (data['Close'].shift(1) - data['Low']).rolling(window=n).sum() * 100
    
    return ar, br

def load_ttm_eps(ticker, end_date):

    subdir='data'
    file_name = 'eps.csv'
     # Construct the path to the CSV file relative to the script directory
    file_path = os.path.join('..', '..', '..', subdir, file_name)

    # Check if CSV file exists
    if os.path.exists(file_path):
        all_eps_data = pd.read_csv(file_path, parse_dates=['Date'])
        ticker_data = all_eps_data[all_eps_data['Ticker'] == ticker]
        
        if not ticker_data.empty:
            last_date = ticker_data['Date'].max()
            if (pd.to_datetime(end_date) - pd.to_datetime(last_date)) <= timedelta(days=90):
                print(f"Loading EPS data for {ticker} from CSV.")
                return ticker_data.set_index('Date')[['TTM EPS']]

    print(f"Downloading latest EPS data for {ticker}.")
    
    chromedriver_path = ChromeDriverManager().install()

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    # Set up the service
    service = Service(executable_path=chromedriver_path)

    # Create the WebDriver instance
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    name = yf.Ticker(ticker).info.get('shortName').split()[0]
    url = f'https://www.macrotrends.net/stocks/charts/{ticker}/{name}/pe-ratio'
    driver.get(url)
    
    wait = WebDriverWait(driver, 2)
    table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'table')))
    headers = [header.text for header in table.find_elements(By.TAG_NAME, 'th')][-4:]
    rows = []
    
    for row in table.find_elements(By.TAG_NAME, 'tr'):
        row_data = [val.text for val in row.find_elements(By.TAG_NAME, 'td')]
        if row_data:
            rows.append(row_data)
    
    df = pd.DataFrame(rows, columns=headers)
    df = df.iloc[1:]  # Remove the first row
    driver.quit()
    
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert EPS to numeric, removing the '$' sign
    df['TTM EPS'] = df['TTM Net EPS'].str.replace('$', '').astype(float)
    
    # Add ticker column
    df['Ticker'] = ticker
    
    # Update the CSV file
    if os.path.exists(file_path):
        all_eps_data = pd.read_csv(file_path, parse_dates=['Date'])
        all_eps_data = all_eps_data[all_eps_data['Ticker'] != ticker]  # Remove old data for this ticker
        all_eps_data = pd.concat([all_eps_data, df[['Date', 'Ticker', 'TTM EPS']]])
    else:
        all_eps_data = df[['Date', 'Ticker', 'TTM EPS']]
    
    all_eps_data.to_csv(file_path, index=False)
    
    return df.set_index('Date')[['TTM EPS']]

def calculate_beta(stock_returns, market_returns, window=252):
    """
    Calculate rolling beta for a stock.
    
    Args:
    stock_returns (pd.Series): Daily returns of the stock
    market_returns (pd.Series): Daily returns of the market (e.g., S&P 500)
    window (int): Rolling window for beta calculation (default is 252 trading days, or 1 year)
    
    Returns:
    pd.Series: Rolling beta values
    """
    # Align the data
    returns = pd.DataFrame({'stock': stock_returns, 'market': market_returns})
    returns = returns.dropna()

    # Calculate rolling covariance and market variance
    rolling_cov = returns['stock'].rolling(window=window).cov(returns['market'])
    rolling_var = returns['market'].rolling(window=window).var()

    # Calculate and return beta
    beta = rolling_cov / rolling_var

    # Fill NaN values in beta
    beta = beta.ffill().bfill()

    first_valid_beta = beta.first_valid_index()
    if first_valid_beta:
        # Fill all NaN values before the first valid beta with the first valid beta value
        beta.loc[:first_valid_beta] = beta.loc[first_valid_beta]

    return beta

def load_stock_data(ticker, start_date, end_date):
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Initialize dictionary to store features
    features = {'Close': data['Close']}

    try:


        # Download BTC data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)


        # Add BTC close price as a feature
        features['BTC'] = btc_data['Close']
    except Exception as e:
        print(f"Error downloading BTC: {e}")
    
    try:
        eth_data = yf.download('ETH-USD', start=start_date, end=end_date)['Close']
        eth_data = eth_data.reindex(features['Close'].index)
        features['ETH'] = eth_data

    except Exception as e:
        print(f"Error loading ETH data: {e}")
    try:
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        
        # Download market data (S&P 500)
        market_data = yf.download('^GSPC', start=start_date, end=end_date)
        market_data['Returns'] = market_data['Close'].pct_change()

        # Calculate beta
        beta = calculate_beta(data['Returns'], market_data['Returns'], window=100)
        features['Beta'] = beta
    except Exception as e:
        print(f"Error calculating Beta: {e}")

    try:
        # Calculate stock-specific volatility index
        stock_volatility = data['Returns'].rolling(window=100).std() * np.sqrt(100)
        stock_volatility = stock_volatility.ffill().bfill()
        features['Vix'] = stock_volatility
    except Exception as e:
        print(f"Error calculating Volatility: {e}")

    try:
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['MACD'] = macd
        features['Signal'] = signal

        # New MACD signal feature
        features['MACD_signal'] = pd.Series(np.where(exp1 > exp2, 1, np.where(exp2 > exp1, -1, 0)), index=data.index)

        # Corrected MACD diff feature
        price_diff_sign = np.sign(data['Close'] - data['Open'])
        exp1_diff_sign = np.sign(exp1 - exp1.shift(1))
        features['MACD_diff'] = pd.Series(
            np.where((price_diff_sign > 0) & (exp1_diff_sign < 0), -1,
                     np.where((price_diff_sign < 0) & (exp1_diff_sign > 0), 1, 0)),
            index=data.index
        ).fillna(0)
        
       
    except Exception as e:
        print(f"Error calculating MACD: {e}")

    try:
        # Calculate daily Turnover Rate
        shares_outstanding = yf.Ticker(ticker).info['sharesOutstanding']
        features['TrunOver'] = data['Volume'] / shares_outstanding
    except Exception as e:
        print(f"Error calculating Turnover Rate: {e}")

    try:
        # Calculate daily Stock Amplitude
        features['Amplitude'] = (data['High'] - data['Low']) / data['Open']
    except Exception as e:
        print(f"Error calculating Amplitude: {e}")
    """
    try:
        # Calculate EPS
        ttm_eps_data = load_ttm_eps(ticker, end_date)
        ttm_eps_data = ttm_eps_data.sort_index()
        
        # Merge the datasets
        combined_data = data.join(ttm_eps_data, how='left')
        
        # Find the last available TTM EPS before or on the start date
        last_available_eps = ttm_eps_data[ttm_eps_data.index <= start_date]['TTM EPS'].iloc[-1] if not ttm_eps_data[ttm_eps_data.index <= start_date].empty else None
        
        # Fill missing values with the last available EPS before or on start_date
        combined_data['TTM EPS'] = combined_data['TTM EPS'].fillna(last_available_eps)
        
        # Forward fill remaining NaN values
        combined_data['TTM EPS'] = combined_data['TTM EPS'].ffill()
        
        # Calculate daily PE ratio
        features['PE'] = combined_data['Close'] / combined_data['TTM EPS']
    except Exception as e:
        print(f"Error calculating PE Ratio: {e}")
    """
    try:
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['RSI'] = rsi
        
        # New RSI signal feature
        rsi_yesterday = rsi.shift(1)
        features['RSI_signal'] = pd.Series(
            np.where(rsi > 80, -1,
                     np.where(rsi < 20, 1,
                              np.where((rsi > 50) & (rsi_yesterday <= 50), 0.5,
                                       np.where((rsi <= 50) & (rsi_yesterday > 50), -0.5, 0)))),
            index=data.index
        ).fillna(0)
        
        # Fill NaN values in RSI
        features['RSI'] = features['RSI'].bfill()
    except Exception as e:
        print(f"Error calculating RSI and RSI_signal: {e}")

    try:
        # Calculate Bollinger Bands  
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        features['Upper_BB'] = sma + (std * 2)
        features['Lower_BB'] = sma - (std * 2)
        features['Upper_BB'] = features['Upper_BB'].ffill().bfill()
        features['Lower_BB'] = features['Lower_BB'].ffill().bfill()
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")

    try:
        # Calculate Moving Averages
        features['SMA_50'] = data['Close'].rolling(window=50).mean()
        features['SMA_200'] = data['Close'].rolling(window=200).mean()
        features['SMA_50'] = features['SMA_50'].ffill().bfill()
        features['SMA_200'] = features['SMA_200'].ffill().bfill()
    except Exception as e:
        print(f"Error calculating Moving Averages: {e}")

    try:
        # Calculate On-Balance Volume (OBV)
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        features['OBV'] = obv
    except Exception as e:
        print(f"Error calculating OBV: {e}")

    try:
        # Market Sentiment: VIX (Fear Index)
        vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
        features['VIX'] = vix
    except Exception as e:
        print(f"Error fetching VIX: {e}")

    # Create DataFrame from features
    re = pd.DataFrame(features)
    
    # Handle NaN values in Beta if it exists
    if 'Beta' in re.columns:
        first_valid_beta = re['Beta'].first_valid_index()
        if first_valid_beta:
            # Add one more line at the top of beta using second day's date
            second_day = re.index[1]
            re.loc[second_day, 'Beta'] = re.loc[first_valid_beta, 'Beta']
            
            # Interpolate between the second day and the first valid beta
            re.loc[second_day:first_valid_beta, 'Beta'] = re.loc[second_day:first_valid_beta, 'Beta'].interpolate()
            
            # Fill the first day with the second day's value
            re.loc[re.index[0], 'Beta'] = re.loc[second_day, 'Beta']

    re = re.ffill().bfill()
    return re

def load_multiple_stock_data(tickers, start_date, end_date):
    all_stock_data = {}
    for ticker in tickers:
        stock_data = load_stock_data(ticker, start_date, end_date)
        all_stock_data[ticker] = stock_data
    return all_stock_data