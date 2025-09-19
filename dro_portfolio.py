# mean-variance portfolio allocation
import pandas as pd
import numpy as np
import math
import cvxpy as cp

import time

from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
import torch
from scipy.stats import chi2, norm, multivariate_normal
from sklearn.neighbors import KernelDensity

from src.portfolio_model import mv_portfolio

import matplotlib.pyplot as plt
import seaborn as sns

#sample_num = 20
# base case: a


def DRO_compare(sample_num, asset_num):
    eval_num = 1
    choice_num = 4
    lambda_1 = 1
    lambda_2 = 2


    # dro_coefs = [1, 3, 5]
    # kfold_num = [2, 5, 10]


    labels = ['Oracle', 'OIC', '5-CV']


    def DGP(asset_num):
        asset_num0 = int(asset_num / 2)
        return_mean = np.random.uniform(0, 4, asset_num)
        cov_comp = np.random.rand(asset_num0, asset_num0) * 0.5
        return_cov = np.zeros((asset_num0, asset_num0))
        return_cov = np.dot(cov_comp, cov_comp.transpose())

        cov_comp = np.random.rand(asset_num0, asset_num0) * 0.5
        return_cov0 = np.zeros((asset_num0, asset_num0))    
        return_cov0 = np.dot(cov_comp, cov_comp.transpose())

        cov_all = np.zeros((asset_num, asset_num))
        cov_all[0:asset_num0, 0:asset_num0] = return_cov
        #= return_cov
        cov_all[asset_num0:, asset_num0:] = return_cov0

        return return_mean, cov_all

    # hyperparameter list
    rho_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return_mean, cov_all = DGP(asset_num)

    mv_model = mv_portfolio(return_mean, cov_all, lambda_1, lambda_2)
    data = mv_model.generate(sample_num)




    loop_num = 50
    outer_num = 10

    model_perform = np.zeros((loop_num, choice_num * len(rho_list)))
    model_avg = np.zeros((outer_num,  choice_num * len(rho_list)))

    final_perform_mean = np.zeros(choice_num * len(rho_list))
    final_perform_std = np.zeros(choice_num * len(rho_list))

    dro_oic_time = 0
    dro_cv_time = 0
    for i in range(outer_num):
        
        for k in range(loop_num):
            print(i, k)
            saa_sol = mv_model.saa_sol(data)
            model_perform[k][0] = mv_model.true_eval(saa_sol)
            model_perform[k][1] = mv_model.sample_eval(saa_sol, data)
            
        
            model_perform[k][2] = mv_model.saa_correction(saa_sol, data) +  model_perform[k][1]

            model_perform[k][3] = mv_model.CV_performance(data, 5, 'saa')


            for j, rho in enumerate(rho_list[1:]):
                mv_model.ambiguity_coef = rho
                dro_sol = mv_model.chi2_DRO_sol(data)
                model_perform[k][choice_num * (j + 1) + 0] = mv_model.true_eval(dro_sol)
                model_perform[k][choice_num * (j + 1) + 1] = mv_model.sample_eval(dro_sol, data)

                model_perform[k][choice_num * (j + 1) + 2] = model_perform[k][choice_num * j + 1] + mv_model.saa_correction(dro_sol, data)
                model_perform[k][choice_num * (j + 1) + 3] = mv_model.CV_performance(data, 5, 'chi2dro')

        
        model_avg[i] = np.mean(model_perform, axis = 0)

    final_perform_mean = np.mean(model_avg, axis = 0)
    final_perform_std = np.std(model_avg, axis = 0)


    marker_choice = ['*', 'o', 'v', '1', '2', '3']
    colors=['#4daf4a', '#ff7f00', '#984ea3', '#FFD43B', '#a65628', '#f781bf', '#e41a1c', '#377eb8']

    model_name = ['Oracle', 'Sample', 'OIC', '5-CV']
    fig = plt.figure()
    for i in range(choice_num):
        model_index = [4* k + i for k in range(len(rho_list))]
        plt.plot(rho_list, final_perform_mean[model_index], label = model_name[i], marker = marker_choice[i], color = colors[i], linewidth = 1) 
        #plt.errorbar(x = rho_list, y = final_perform_mean[model_index], yerr = final_perform_std[model_index] / math.sqrt(outer_num), color = colors[i], capsize = 2)

    plt.legend(loc = 'upper left')
    plt.xlabel('rho')
    plt.ylabel('avg performance')
    plt.savefig(f'figures/portfolio/DRO_{sample_num}_{asset_num}.pdf')

if __name__ == '__main__':
    DRO_compare(100, 40)










