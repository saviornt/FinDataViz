import datetime as dt
import logging
from datetime import timedelta

import pandas as pd
import pandas_datareader as pdr
import pandas_gbq as pb
import requests
import talib as ta
from google.cloud import bigquery, bigquery_datatransfer
from sklearn.impute import KNNImputer

import config
import fred_parameters
import census_parameters
import market_parameters
from census_parameters import census_dictionary

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GetData:

    @staticmethod
    def get_security_info(symbol, start_date, end_date):
        start_date = CleanData.get_previous_days(start_date, 45)
        df = pdr.get_data_yahoo(symbols=symbol, start=start_date, end=end_date)
        df['Avg Price'] = ta.AVGPRICE(df['Open'], df['High'], df['Low'], df['Close'], )
        df['SMA'] = ta.SMA(df['Close'], timeperiod=5)
        df['EMA'] = ta.EMA(df['Close'], timeperiod=5)
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
        df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df = df.iloc[45 - 14:]
        return df

    @staticmethod
    def get_fred_data(code, start_date, end_date):
        start = dt.datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y-%m-%d")
        end = dt.datetime.strptime(end_date, "%m/%d/%Y").strftime("%Y-%m-%d")
        df = pdr.DataReader(code, 'fred', start, end)
        df = CleanData.cleanup_dataframe_ML(df)
        return df

    @staticmethod
    def get_census_data(census_code, start_date, end_date):
        api_key = config.census_api_key
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

    @staticmethod
    def get_market_data(start_date, end_date):
        market_data = {}
        ticker_symbols = market_parameters.get_market_symbols()
        for symbol in ticker_symbols:
            df = GetData.get_security_info(symbol, start_date, end_date)
            market_data.update({symbol: df})
        return market_data

    @staticmethod
    def get_total_fred_data(start_date, end_date):
        fred_data = {}
        fred_codes = fred_parameters.get_fred_codes()
        for code in fred_codes:
            df = GetData.get_fred_data(code, start_date, end_date)
            fred_data.update({code: df})
        return fred_data

    @staticmethod
    def get_total_census_data(start_date, end_date):
        census_data = {}
        census_codes = census_parameters.get_census_codes()
        for census_code in census_codes:
            df = GetData.get_census_data(census_code, start_date, end_date)
            census_data.update({census_code: df})
        return census_data


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
        quarter = ''
        month = month.upper()  # Validate and correct input to uppercase
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

    @staticmethod
    def get_previous_days(date_string, amount_of_days):
        date_string = dt.datetime.strptime(date_string, '%m/%d/%Y').date()
        date_string = date_string - timedelta(days=amount_of_days)
        date_string = date_string.strftime('%m/%d/%Y')
        return date_string


class BigQueryMethods:
    """
    TODO Create DOCSTRING
    """

    @staticmethod
    def create_dataset(dataset_name, dataset_location):
        client = bigquery.Client()
        dataset_id = "{}.dataset_name".format(client.project)
        dataset = bigquery.Dataset(dataset_id)
        dataset_location = dataset_location
        try:
            dataset = client.create_dataset(dataset, timeout=30)
        except BaseException as e:
            logging.error(e)

        # Set a table expiration timeframe to never expire
        dataset = client.get_dataset(dataset_id)
        dataset.default_table_expiration_ms = 'Never'
        dataset = client.update_dataset(dataset, ["default_table_expiration_ms"])
        return

    @staticmethod
    def copy_dataset(source_dataset_name, destination_dataset_name):
        project_id = config.project_id
        transfer_client = bigquery_datatransfer.DataTransferServiceClient()

        source_project_id = destination_project_id = project_id
        source_dataset_id = source_dataset_name
        destination_dataset_id = destination_dataset_name

        transfer_config = bigquery_datatransfer.TransferConfig(
            destination_dataset_id=destination_dataset_id,
            display_name="Your Dataset Copy Name",
            data_source_id="cross_region_copy",
            params={
                "source_project_id": source_project_id,
                "source_dataset_id": source_dataset_id,
            },
            schedule="every 24 hours",
        )
        transfer_config = transfer_client.create_transfer_config(
            parent=transfer_client.common_project_path(destination_project_id),
            transfer_config=transfer_config,
        )
        return

    @staticmethod
    def list_datasets():
        client = bigquery.Client()
        datasets = list(client.list_datasets())
        return datasets

    @staticmethod
    def delete_dataset(dataset_name, ):
        project_id = config.project_id
        client = bigquery.Client()
        dataset_id = '{}.{}'.format(project_id, dataset_name)
        client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=True)
        return

    @staticmethod
    def create_schema_fields(number_of_fields: int):
        fields = []
        for f in range(number_of_fields):
            field_name = input('Field name: ')
            field_type = input('Field type: ')
            field_mode = input('Field mode: ')
            field_mode = "mode={}".format(field_mode)
            field = "{}, {}, {}".format(field_name, field_type, field_mode)
            fields.append([field])
        return fields

    @staticmethod
    def create_schema(fields: list):
        schema = []
        for f in fields:
            schema.append = [bigquery.SchemaField(f)]
        return schema

    @staticmethod
    def create_table(dataset_name, table_name, schema: list = None):
        client = bigquery.Client()
        project_id = config.project_id
        table_id = "{}.{}.{}".format(project_id, dataset_name, table_name)
        if schema is None:
            table = bigquery.Table(table_id)
        else:
            table = bigquery.Table(table_id, schema=schema)

        table = client.create_table(table)
        return

    @staticmethod
    def create_table_from_dataframe(dataframe, dataset_name, table_name, table_expiration='Never',
                                    table_description=''):
        project_id = config.project_id
        table_id = '{}.{}'.format(dataset_name, table_name)
        pb.to_gbq(dataframe, table_id, project_id)

        if table_description is not None:
            BigQueryMethods.set_table_description(dataset_name, table_name, table_description)

        if table_expiration is not 'Never':
            BigQueryMethods.set_table_expiration(dataset_name, table_name, table_expiration)

        return

    @staticmethod
    def set_table_description(dataset_name, table_name, table_description):
        client = bigquery.Client()
        project_id = config.project_id
        dataset_ref = bigquery.DatasetReference(project_id, dataset_name)
        table_ref = dataset_ref.table(table_name)
        table = client.get_table(table_ref)
        table.description = table_description
        table = client.update_table((table, ["description"]))
        return

    @staticmethod
    def set_table_expiration(dataset_name, table_name, table_expiration='Never'):
        client = bigquery.Client()
        project = client.project
        dataset_ref = bigquery.DatasetReference(project, dataset_name)
        table_ref = dataset_ref.table(table_name)
        table = client.get_table(table_ref)
        if table_expiration is not 'Never':
            table_expiration = dt.datetime.now(dt.timezone.utc) + timedelta(days=5)
        table.expires = table_expiration
        table = client.update_table(table, ["expires"])
        return

    @staticmethod
    def copy_table(source_dataset_name, source_table_name, destination_dataset_name, destination_table_name):
        project_id = config.project_id
        client = bigquery.Client()
        source_table = '{}.{}.{}'.format(project_id, source_dataset_name, source_table_name)
        destination_table = '{}.{}.{}'.format(project_id, destination_dataset_name, destination_table_name)
        job = client.copy_table(source_table, destination_table)
        job.result()
        return

    @staticmethod
    def delete_table(dataset_name, table_name):
        project_id = config.project_id
        client = bigquery.Client()
        table_id = '{}.{}.{}'.format(project_id, dataset_name, table_name)
        client.delete_table(table_id, not_found_ok=True)
        return

    @staticmethod
    def append_table_data(df, dataset_name, table_name):
        project_id = config.project_id
        pb.to_gbq(df, dataset_name, table_name, if_exists='append')
        return

    @staticmethod
    def get_table_data(dataset_name, table_name, sql_statement):
        project_id = config.project_id
        df = pb.read_gbq(sql_statement, project_id)
        return df


class Graphs:
    @staticmethod
    def create_stock_market_graph(security_dataframe, symbol):
        security_dataframe.reset_index(inplace=True)
        price_min_range = security_dataframe['Avg Price'].min()
        price_min_range = price_min_range - (price_min_range * 0.05)

        price_max_range = security_dataframe['Avg Price'].max()
        price_max_range = price_max_range + (price_max_range * 0.05)

        # Create subplots and grid sizes
        fig_prices = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                   subplot_titles=(
                                       'Price Movement', 'Volume of Shares Traded & Relative Strength Index',
                                       'Average Directional Index'), row_width=[0.3, 0.3, 0.3],
                                   specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])

        # Plot the OHLC Candlesticks
        fig_prices.add_trace(go.Candlestick(x=security_dataframe['Date'],
                                            open=security_dataframe['Open'],
                                            high=security_dataframe['High'],
                                            low=security_dataframe['Low'],
                                            close=security_dataframe['Close'],
                                            name='OHLC Candlesticks'), row=1, col=1, secondary_y=False)

        # Add the Moving Averages to the top graph
        fig_prices.add_scatter(x=security_dataframe['Date'],
                               y=security_dataframe['SMA'], mode='lines',
                               name='Simple Moving Average', row=1, col=1)

        fig_prices.add_scatter(x=security_dataframe['Date'],
                               y=security_dataframe['EMA'], mode='lines',
                               name='Exponential Moving Average', row=1, col=1)

        # Add the volume bars and RSI on the second row without adding it to the legend

        fig_prices.add_trace(go.Bar(x=security_dataframe['Date'], y=security_dataframe['Volume'],
                                    showlegend=False), row=2, col=1, secondary_y=False)

        # Add the RSI indicator to the third row without adding to the legend
        fig_prices.add_trace(go.Scatter(x=security_dataframe['Date'], y=security_dataframe['RSI'],
                                        showlegend=False), row=2, col=1, secondary_y=True)

        # Add the ADX indicator to the third row without adding to the legend
        fig_prices.add_trace(go.Scatter(x=security_dataframe['Date'], y=security_dataframe['ADX'],
                                        showlegend=False), row=3, col=1)

        fig_prices.update_layout(title='Stock Market History for {}'.format(symbol.upper()), yaxis_title='Price Range',
                                 autosize=True)

        fig_prices.update(layout_xaxis_rangeslider_visible=False)
        fig_prices.show()
        return
