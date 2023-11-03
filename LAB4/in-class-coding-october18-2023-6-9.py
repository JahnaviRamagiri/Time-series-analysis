import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv"
from statsmodels.tsa.seasonal import STL, seasonal_decompose

data = pd.read_csv(url)
Temp = data['#Passengers']
Temp = pd.Series(np.array(Temp), index=pd.date_range('1981-01-01'
                                                     ,periods=len(Temp),
                                                     freq='m'),
                 name='Temp')
Temp_df  = pd.DataFrame(Temp)
Temp_df.plot()
plt.show()
# ===================
# STL decomposition
# =================
STL = STL(Temp)
res = STL.fit()
fig = res.plot()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

def str_trend_seasonal(T,S,R):
    F = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(R+T)))
    print(f'Strength of trend for the raw data is {100 * F:.3f}%')

    FS = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S + R)))
    print(f'Strength of seasonality for the raw data is {100 * FS:.3f}%')

str_trend_seasonal(T,S,R)