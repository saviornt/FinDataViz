import pandas as pd
import classes
import config
import fred_dictionary as fd

config.set_env_variables()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

project_id = config.project_id
census_api_key = config.census_api_key
start_date = "1/1/2002"
end_date = "1/1/2022"
symbol = "SPY"
census_code = "vip"
reports = "monthly"
start_year = "2002"
end_year = "2022"
fred_dictionary = fd.fred_dictionary
fred_codes = fred_dictionary.keys()

stock_data = classes.GetData.get_security_info(symbol, start_date, end_date)
fred_data = classes.GetData.get_fred_data(fred_codes, start_date, end_date)
census_vip_data = classes.GetData.get_census_data(census_code, reports, start_year, end_year, census_api_key)