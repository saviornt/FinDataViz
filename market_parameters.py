symbol_dictionary = {
    'DJIA': 'Dow Jones Industrial Average Index ETF',
    'SPY': 'S&P-500 Index ETF',
    'QQQ': 'NASDAQ-100 Index ETF',
    'VTI': 'Vanguard Total Stock Market Index',
    'VIX': 'CBOE Volatility Index'
}


def get_market_symbols():
    symbols = symbol_dictionary.keys()
    return symbols
