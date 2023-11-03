import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import make_classification, make_regression

X , y = make_classification(n_samples= 1000,
                        n_features = 100,
                        n_informative = 95,
                        n_repeated = 5,
                        n_redundant = 0,
                        n_classes = 4,
                        random_state = 6313 )

X_1 = pd.DataFrame(X, columns = [f'feature{x}' for x in range(1, X.shape[1]+1)])
y_1 = pd.DataFrame(y, columns=['target'])
df = pd.concat([X_1, y_1], axis =1)
vif_data1 = pd.DataFrame()
vif_data1['features'] = df.columns
vif_data1['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
print(vif_data1)

np.random.seed(6313)

N = 1000
mean = 0
std = 1

x1 = np.random.normal(mean, std, N)
x2 = np.random.normal(mean, std, N)
x3 = 3*x2#np.random.normal(mean, std, N)
x4 = 2*x1#np.random.normal(mean, std, N)

X = np.vstack((x1,x2,x3,x4)).T

# =====
# SVD
# =====
U, S, V = np.linalg.svd(X)
print('Singular Values of X', S)
print(f'Condition number of X {np.linalg.cond(X)}')

# =====
# VIF
# =====

X = pd.DataFrame(X, columns = ['A','B','C','D'])

vif_data1 = pd.DataFrame()
vif_data1['features'] = X.columns
vif_data1['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data1)

