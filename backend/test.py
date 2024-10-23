import csv

import pandas as pd

tariff_file = 'cennik.csv'
df = pd.read_csv(tariff_file, delimiter=';', encoding='utf-8', on_bad_lines='skip')

print(df.head(10))