from datetime import datetime, timedelta

from yahoo_fin import stock_info as si


def get_info(ticker, days_cnt=False):
    check_ticker = is_sp500(ticker)
    if check_ticker == ticker:
        if days_cnt:
            now = datetime.now()
            start_date = now - timedelta(days=days_cnt)
            return si.get_data(ticker, start_date)
        return si.get_data(ticker)
    else:
        return check_ticker


def is_sp500(ticker):
    return 'Ticker not in S&P 500. Please try another ticker' if ticker not in si.tickers_sp500() else ticker
