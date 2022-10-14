import pandas as pd
import pandas_datareader as pdr
import talib as ta
import datetime as dt
from sklearn.impute import KNNImputer


class Securities:

    @staticmethod
    def cleanup_dataframe(df):
        df = df.dropna()
        return df

    @staticmethod
    def cleanup_dataframe_ML(df):
        # First we'll clean the majority of the cells using linear interpolation
        df = df.interpolate(method='linear')

        # Next, we'll copy the index to a list since KNN Imputer drops the index
        df_index_list = df.index.tolist()

        # Next we'll use the KNNImputer to clean the rest of the NaN values.
        # Note: We don't use MinMaxScaler since we're dealing with financials.
        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        df = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

        # Finally we add the Date list back into the dataframe and set it to the index.
        df['Date'] = df_index_list
        df = df.set_index('Date')

        return df

    @staticmethod
    def get_security_info(symbol, start_date, end_date):
        df = pdr.get_data_yahoo(symbols=symbol, start=start_date, end=end_date)
        df['Avg Price'] = ta.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'], )
        df['SMA'] = ta.SMA(df['Close'], timeperiod=5)
        df['EMA'] = ta.EMA(df['Close'], timeperiod=5)
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
        df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df = Securities.cleanup_dataframe(df)
        return df


def get_fred_data(code, start_date, end_date):
    start = dt.datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    end = dt.datetime.strptime(end_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    df = pdr.DataReader(code, 'fred', start, end)
    return df
