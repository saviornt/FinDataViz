import classes, config
import pandas as pd
import fred_dictionary as fd

config.set_env_variables()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

start_date = "1/1/2002"
end_date = "1/1/2022"
symbol = "SPY"
fred_dictionary = fd.fred_dictionary
fred_codes = fred_dictionary.keys()



stock_data = classes.Securities.get_security_info(symbol, start_date, end_date)
