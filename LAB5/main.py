from modules import arma, statistics as st, utils
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(6313)
    utils.print_title("GPAC Table Implementation")
    data_samples = int(input("Enter the number of data Samples "))
    mean_WN = float(input("Enter the mean of White Noise "))
    var_WN = float(input("Enter the variance of White Noise "))
    na = int(input("Enter AR order "))
    nb = int(input("Enter MA Order "))
    AR_coef = []
    MA_coef = []
    for i in range(1, na+1):
        AR_coef.append(float(input(f"Enter AR coefficient : a{i} ")))

    for i in range(1, nb+1):
        MA_coef.append(float(input(f"Enter MA coefficient : b{i} ")))


    # ARMA Process
    utils.print_title("ARMA Process", pattern="`")
    arma_process = arma.get_process(AR_coef, MA_coef)
    print(arma_process)
    print("Is this a stationary process : ", arma_process.isstationary)
    # Generate ARMA Process dataset
    mean_y = mean_WN*(1 + np.sum(MA_coef)) / (1 + np.sum(AR_coef))
    y = arma_process.generate_sample(data_samples, scale=np.sqrt(var_WN)) + mean_y

    # Theoritical ACF
    utils.print_title("Theoritical ACF", pattern="`")
    max_lag = 20
    ry = arma_process.acf( lags = max_lag)
    print(np.round(ry, 2))
    ry_ = ry[::-1]
    ry = np.concatenate((np.reshape(ry_,max_lag), ry[1:]))


    # GPAC
    utils.print_title("GPAC Table", pattern="`")
    gpac = st.get_gpac(ry)
    sns.heatmap(gpac, xticklabels = list(range(1,7)), annot=True, cmap= "RdBu")
    plt.title(f"GPAC : ARMA({na},{nb}) AR {AR_coef} MA {MA_coef}")
    plt.show()
    print(gpac)

    # ACF and PACF
    st.ACF_PACF_Plot(y, max_lag)



