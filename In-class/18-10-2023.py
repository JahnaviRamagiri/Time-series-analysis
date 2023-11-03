import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from LAB4 import utils

from statsmodels.tsa.seasonal import STL

data = pd.read_csv("AirPassengers.csv")
print()
data.plot()
plt.show()
temp = pd.Series()
# STL Decomposition
utils.print_title("STL Decomposition")
STL = STL(data["#Passengers"])
res = STL.fit()


T = res.trend
R = res.resid
S = res.seasonal



Ft = max(0,1-np.var(R)/np.var(T+R))
print(T,R,S,Ft)