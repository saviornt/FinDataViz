import local_vars

project_id = "fin-data-viz"
census_api_key = local_vars.census_api_key

def set_env_variables():
    import os
    json_path = local_vars.json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
