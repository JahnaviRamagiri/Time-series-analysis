import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
sys.path.append('../toolbox')
from toolbox import autocorr, Cal_autocorr,ACF_PACF_Plot
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split

df = pd.read_csv("AirPassengers.csv", index_col="Month", parse_dates=True)
y = df["#Passengers"]
lags = 38
ACF_PACF_Plot(y, lags)
yt, yf = train_test_split(y, shuffle= False, test_size=0.2)

# =============================
# = SES Method
# ============================
holtt = ets.ExponentialSmoothing(yt,trend=None,damped=False,seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for simple exponential smoothing is ", MSE)
#===============
# Plotting area
# ===============
fig, ax = plt.subplots()
ax.plot(yt,label= "Train Data")
ax.plot(yf,label= "Test Data")
ax.plot(holtf,label= "Simple Exponential Smoothing")
plt.legend(loc='upper left')
plt.title('Simple Exponential Smoothing- Air Passengers')
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.show()

# =============================
# = Holt Linear Method
# ============================
holtt = ets.ExponentialSmoothing(yt,trend='mul',damped=True,seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)
MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for double exponential smoothing is ", MSE)

fig, ax = plt.subplots()
ax.plot(yt,label= "Train Data")
ax.plot(yf,label= "Test Data")
ax.plot(holtf,label= "Holt's Linear Trend Method")
plt.legend(loc='upper left')
plt.title("Holt's Linear Trend Method")
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.show()


# =============================
# = Holt-Winter Method
# ============================
holtt = ets.ExponentialSmoothing(yt,trend='mul',damped=True,seasonal='mul').fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for holt-winter method is ", MSE)
fig, ax = plt.subplots()
ax.plot(yt,label= "train")
ax.plot(yf,label= "test")
ax.plot(holtf,label= "Holt-Winter Method")

plt.legend(loc='upper left')
plt.title('Holt-Winter Method')
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.show()

# N = 10000
# mean = 0
# std = 1
# # ==================================
# # Create an Autoregressive process
# # y(t) - 0.9y(t-1) = e(t)
# # ================================
#
# np.random.seed(6313)
# e = np.random.normal(mean, std, size = N)
#
# y = np.zeros(len(e))
# for t in range(len(e)):
#     if t ==0:
#         y[t] = e[t]
#
#     else:
#         y[t] = 0.9*y[t-1] + e[t]
#
#
#
# date = pd.date_range(start = '2000-01-01',
#                      periods = len(y),
#                      freq = 'D')
# df = pd.DataFrame(data = y , columns=['y'], index = date)
#
# df.plot()
# plt.grid()
# plt.xlabel('date')
# plt.ylabel('Mag.')
# plt.title('White noise')
# plt.tight_layout()
# plt.show()
# lag = 20
# ry = Cal_autocorr(y, lag  , 'white noise')
#
# a = [1, .5, .25, .125, .0625]
# b = a[::-1]
# c = b + a[1:]
#
# (markers, stemlines, baseline) = plt.stem(c, markerfmt='o')
# plt.setp(markers, color = 'red', marker = 'o')
# plt.setp(baseline, color='grey', linewidth=2, linestyle='-')
# m = 1.96/np.sqrt(N)
# plt.axhspan(-m, m,alpha = 0.2, color = 'blue')
# plt.show()