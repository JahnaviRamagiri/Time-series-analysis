from modules import utils, arma
import statsmodels.api as sm

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(6313)
    utils.print_title("LM Algorithm")
    data_samples = int(input("Enter the number of data Samples "))
    mean_WN = float(input("Enter the mean of White Noise "))
    var_WN = float(input("Enter the variance of White Noise "))
    na = int(input("Enter AR order "))
    nb = int(input("Enter MA Order "))
    # delta = float(input("Enter Delta "))
    # max_iter = int(input("Enter Max iterations for LM Algorithm "))
    # thres = float(input("Enter Epsilon (e) "))
    # mu_max = float(input("Enter Mu Max"))

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
    theta_true = AR_coef + MA_coef
    # LM ALgorithm
    utils.print_title("LM Algorithm", pattern="`")
    lm_output = utils.get_LM(na, nb, y, data_samples, mean_WN, var_WN)
    # lm_output = utils.get_LM(na, nb, y, data_samples, delta, max_iter, thres,  mu_max)
    # print(lm_output)

    sse, theta, sd, cov_matrix = lm_output
    print("Estimated Theta Values", theta)
    print("True Theta Values", theta_true)
    print("Standard Deviation", sd)
    print("Covariance Matrix", cov_matrix )

    utils.print_title("Confidence Interval", pattern="`")
    con_interval = utils.get_CI(theta, cov_matrix)
    for i in range(len(theta)):
        if i < na:
            print(con_interval[0][i], f"< a{i+1} <" , con_interval[1][i])
        else:
            print(con_interval[0][i], f"< b{i + 1} <", con_interval[1][i])


    # utils.print_title("Variance", pattern="`")

    utils.print_title("Contruct generated ARMa Process")
    est_an = theta[:na]
    est_bn = theta[na:]
    est_arma_process = arma.get_process(est_an, est_bn)
    mean_y = mean_WN * (1 + np.sum(est_bn)) / (1 + np.sum(est_an))
    est_y = est_arma_process.generate_sample(data_samples, scale=np.sqrt(var_WN)) + mean_y

    error = y - est_y
    err_var = np.var(error)
    print("Variance of Error is ", err_var)

    plt.plot(sse)
    plt.xlabel("Iterations")
    plt.ylabel("Sum of Squared Error")
    plt.title("SSE vs Iterations")
    plt.show()

    # utils.print_title("Transfer Function")
    # arparams = np.array(an)
    # maparams = np.array(bn)
    #
    # den = np.r_[1, arparams]
    # num = np.r_[1, maparams]
    # system = signal.TransferFunction(arma_process.a, MA_coef)

    utils.print_title("Phase 2")
    model = sm.tsa.arima.ARIMA(y, order=(na, 0, nb), trend='n').fit()
    print(model.summary())




