import datetime as dt
import pandas as pd
import pandas_datareader as pdr
import requests
import talib as ta
from sklearn.impute import KNNImputer
from census_parameters import census_dictionary


class GetData:

    @staticmethod
    def get_security_info(symbol, start_date, end_date):
        df = pdr.get_data_yahoo(symbols=symbol, start=start_date, end=end_date)
        df['Avg Price'] = ta.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'], )
        df['SMA'] = ta.SMA(df['Close'], timeperiod=5)
        df['EMA'] = ta.EMA(df['Close'], timeperiod=5)
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
        df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df = CleanData.cleanup_dataframe(df)
        return df

    @staticmethod
    def get_fred_data(code, start_date, end_date):
        start = dt.datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y-%m-%d")
        end = dt.datetime.strptime(end_date, "%m/%d/%Y").strftime("%Y-%m-%d")
        df = pdr.DataReader(code, 'fred', start, end)
        return df

    @staticmethod
    def get_census_data(census_code, start_date, end_date, api_key):
        start_month = start_date.dt.datetime.strftime("%m")
        start_year = start_date.dt.datetime.strftime("%Y")
        end_month = end_date.dt.datetime.strftime("%m")
        end_year = end_date.dt.datetime.strftime("%Y")

        reports = (census_dictionary[census_code])[1]
        if reports == 'quarterly':
            start_month = CleanData.month2quarter(start_month)
            end_month = CleanData.month2quarter(end_month)

        base_url = "https://api.census.gov/data/timeseries/eits/{}?".format(census_code)
        param_url = "get=cell_value,time_slot_id,error_data,category_code&for&seasonally_adj&data_type_code&"
        time_url = "time=from+{}-{}+to+{}-{}&key={}".format(start_year, start_month, end_year, end_month, api_key)
        url = base_url + param_url + time_url
        response = requests.request('GET', url)

        df = pd.DataFrame(response.json()[1:], columns=response.json()[0])
        df = df.drop(columns=['time_slot_id', 'error_data', 'category_code', 'seasonally_adj', 'data_type_code'])
        df = CleanData.total_report_date_revenue(df, census_code)
        return df


class CleanData:
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
    def month2quarter(month):
        if month <= 3:
            quarter = 'Q1'
        elif (month <= 6) and (month >= 4):
            quarter = 'Q2'
        elif (month <= 9) and (month >= 5):
            quarter = 'Q3'
        elif (month <= 12) and (month >= 9):
            quarter = 'Q4'
        return quarter

    @staticmethod
    def total_report_date_revenue(df, census_code):
        new_data = {}
        df_duplicates = df[df.duplicated('time')]
        duplicate_rows = 1
        while duplicate_rows > 0:
            try:
                date = df_duplicates['time'].iloc[0]
                df_date = df.loc[df['time'] == date]
                revenue_list = df_date['cell_value'].values.tolist()
                revenue_list = [eval(i) for i in revenue_list]
                rev = int(sum(revenue_list))
                new_data.update({date: rev})
                df_duplicates = df_duplicates[df.time != date]
                duplicate_rows = df_duplicates.duplicated().sum()
            except Exception as e:
                print(e)
                break
        df = pd.DataFrame.from_dict(new_data, orient='index')
        df.index.name = "Date"
        df = df.rename(columns={0: census_code.upper() + " Revenue"})
        return df


class BigQueryMethods:
    """
    TODO Create DOCSTRING
    TODO Create, Copy, Delete Dataset
    TODO Create, Copy, Delete Table in a Dataset
    TODO Load data into table
    TODO Get data from table
    """
