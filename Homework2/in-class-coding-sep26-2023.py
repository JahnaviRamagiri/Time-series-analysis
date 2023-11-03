import statsmodels.api as sm
import pandas as pd
import numpy as np


X = np.array([[1],[2],[3],[4],[5]])
Y = np.array([[2],[4],[5],[4],[5]])
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())







#
#
#
#
# X = np.array([[1],[2],[3],[4],[5]])
# Y = np.array([[2],[4],[5],[4],[5]])
# X = sm.add_constant(X)
# model = sm.OLS(Y,X).fit()
# print(model.summary())

