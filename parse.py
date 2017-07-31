import csv
import numpy as np

def getData():

    list = []

    with open('sangyohi.csv', 'r') as f:
         reader = csv.reader(f)
         for row in reader:
             list.append(row)
    return np.array(list).astype(np.float)
