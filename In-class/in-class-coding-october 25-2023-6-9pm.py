#==========================================
# Simulate the AR(1) model
# y(t) + 0.5y(t-1) = e(t)-------e(t)~WN(0,1)
#=========================================
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\GW\Time series Analysis\toolbox')
from toolbox import Plot_Rolling_Mean_Var, ADF_Cal, kpss_test, difference
np.random.seed(6313)
T = 10000
mean = 0
var = 1

e = np.random.normal(mean, var, T)
# y = np.zeros(len(e))
# for t in range(len(e)):
#     if t==0:
#         y[t]=e[t]
#
#     else:
#         y[t] = -0.5 * y[t - 1] + e[t]
# print(f'Sampled mean is {np.mean(y)}')
# print(f'Sampled var is {np.var(y)}')
# plt.plot(y)
# plt.show()

#==========================================
# Simulate the AR(1) model using dlsim
# y(t) + 0.5y(t-1) = e(t)-------e(t)~WN(0,1)
#=========================================
import sys
sys.path.append('../toolbox')
from toolbox import ADF_Cal, autocorr, Cal_autocorr, Cal_GPAC
from scipy import signal
num = [1, 0,0]
den = [1, -1.5, 0.5]
sys = (num, den, 1)
_,y = signal.dlsim(sys, e)
# print('for loop simulation', y[:5])
y = np.ndarray.flatten(y)
plt.plot()
plt.show()
ry = Cal_autocorr(y,20,'ACF')
Plot_Rolling_Mean_Var(y,'raw data')