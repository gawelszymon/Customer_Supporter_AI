import csv

import pandas as pd

# with open('cennik.csv', newline='') as tariff:
#     tariffreader =  csv.reader(tariff, delimiter=';')
#     rows = list(tariffreader)
    
#     for i in range(2):
#         print(' | \n'.join(rows[i]))
#         print("_____________________________________________________")


tariff_file = '../cennik.csv'
df = pd.read_csv(tariff_file, delimiter=';')

print(df.head(10))
