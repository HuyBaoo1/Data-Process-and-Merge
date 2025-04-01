import pandas as pd

def load_data(file_path, sheet_name=1):
    return pd.read_excel(file_path, sheet_name=sheet_name)