import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import re
import xlsxwriter

filename = "data.txt"
file = open(filename)

arr = []
for line in file:
    p = re.split(', \"[1-9]\": ', line[6: len(line) - 2])
    for rgb in p:
        rgb = rgb[1: len(rgb) - 1]
        print(rgb)
        arr.append(rgb.split(", "))

arr = np.array(arr)
r = list(map(lambda x: float(x), arr[:, 0].tolist()))
g = list(map(lambda x: float(x), arr[:, 1].tolist()))
b = list(map(lambda x: float(x), arr[:, 2].tolist()))

df = pd.DataFrame({'R': r, 'G': g, 'B': b})
writer = pd.ExcelWriter('brightness.xlsx', engine='xlsxwriter')
df.to_excel(writer, index=False)
writer.save()
