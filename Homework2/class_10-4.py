import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_regression



np.random.seed(6313)
mean = 0
std = 1
N = 1000
wn1 = np.random.normal(mean, std, size=N)
wn2 = np.random.normal(mean, std, size=N)
wn3 = np.random.normal(mean, std, size=N)
wn4 = np.random.normal(mean, std, size=N)

plt.plot(wn1)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('WNoise 1')
plt.tight_layout()
plt.show()

x = np.vstack((wn1, wn2, wn3, wn4)).T


## SVD

U, S, V = np.linalg.svd(x)
print("Singular value of x", S) ## Significant
print("Condition Number of x", np.linalg.cond(x)) ## good - no collinearity

## VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

x = pd.DataFrame(x, columns= ['A', 'B','C','D'])

vif_data1 = pd.DataFrame()
vif_data1["features"] = x.columns
vif_data1["VIF"] = [variance_inflation_factor(x.values, i ) for i in range(len(x.columns)) ]
print(vif_data1)



X, y = make_classification(n_samples = 1000,
                       n_features = 100,
                       n_informative = 95,
                       n_repeated = 5,
                           n_redundant= 0,
                       n_classes = 4,
                           random_state= 6313)

U, S, V = np.linalg.svd(X)
print("Singular value of x", S)
print("Condition Number of x", np.linalg.cond(x))

vif_data1 = pd.DataFrame(X)
vif_data1["features"] = X.columns
vif_data1["VIF"] = [variance_inflation_factor(x.values, i ) for i in range(len(X.columns)) ]
print(vif_data1)


