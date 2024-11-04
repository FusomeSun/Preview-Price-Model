from datetime import datetime, timedelta

def is_trading_day(date):
    # Check if it's a weekend
    if date.weekday() >= 5:
        return False
    
    # Basic U.S. stock market holidays (2024)
    holidays = [
        datetime(2024, 1, 1),   # New Year's Day
        datetime(2024, 1, 15),  # Martin Luther King Jr. Day
        datetime(2024, 2, 19),  # Presidents' Day
        datetime(2024, 3, 29),  # Good Friday
        datetime(2024, 5, 27),  # Memorial Day
        datetime(2024, 6, 19),  # Juneteenth
        datetime(2024, 7, 4),   # Independence Day
        datetime(2024, 9, 2),   # Labor Day
        datetime(2024, 11, 28), # Thanksgiving Day
        datetime(2024, 12, 25), # Christmas Day
    ]
    
    return date.date() not in [holiday.date() for holiday in holidays]

def get_previous_trading_day(date):
    previous_day = date - timedelta(days=1)
    while not is_trading_day(previous_day):
        previous_day -= timedelta(days=1)
    return previous_day

def get_nth_previous_trading_day(date, n):
    current_date = date
    for _ in range(n):
        current_date = get_previous_trading_day(current_date)
    return current_date

# Example usage
today = datetime(2024, 10, 8)  # July 15, 2024
trading_days_back = 10

result = get_nth_previous_trading_day(today, trading_days_back)

print(f"Today's date: {today.date()}")
print(f"The {trading_days_back}th previous trading day: {result.date()}")