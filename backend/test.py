import csv

import pandas as pd

tariff_file = 'backend/cennik.csv'
df = pd.read_csv(tariff_file, delimiter=';', encoding='utf-8', on_bad_lines='skip')

print(df['CENA_KLIENT'].head(20))