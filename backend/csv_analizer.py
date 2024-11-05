import re

import pandas as pd


def bad_field_dropper():
    df = pd.read_csv("base/cennik.csv", sep=";")

    try:
        df_e = pd.read_csv("base/cennik.csv")
    except pd.errors.ParserError as e:
        error_msg = str(e)
        match = re.search(r'line (\d+)', error_msg)
        idx = int(match.group(1))
        df = df.drop(idx - 2)
        df.to_csv("base/cennik.csv", sep=";")
        bad_field_dropper()
        
bad_field_dropper()
