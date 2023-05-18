#!/user/tw2837/.conda/envs/tw2837/bin/python
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from scipy.stats import norm
from sklearn.utils import resample
import cvxpy as cp
from scipy.stats import t

from src.nv_model import newsvendor

    

sample_num = 50

mode = 'normal'
def run(sample_num, mode):
    outer_num = 50
    loop_num = 100

    #KFOLD_LIST = [2, 3, 4, 5, 10, sample_num]
    KFOLD_LIST = [5, 10, sample_num]
    BOOTSTRAP_LIST = [10, 50]


    method_num = 4
    choice_num = len(KFOLD_LIST) + len(BOOTSTRAP_LIST) + 5
    error_avg = np.zeros((outer_num, 4 * choice_num))


    print(sample_num, mode)
    for s in range(outer_num):
        print(s)
        error = np.zeros((4 * choice_num, loop_num))

        for i in range(loop_num):
            #print(i)
            nv_model = newsvendor(5, 2, mode)
            data = nv_model.generate(sample_num)

            saa_sol = nv_model.saa(data)
            j = 0
            error[choice_num * j][i] = nv_model.true_eval(saa_sol)
            error[choice_num* j + 1][i] = nv_model.true_eval(saa_sol) - nv_model.empirical_eval(saa_sol, data)
            error[choice_num* j + 2][i] = error[choice_num* j + 1][i] - nv_model.saa_correction(saa_sol, data)
            # for k, fold_val in enumerate(KFOLD_LIST):
            #     error[choice_num * j + 4 + k][i] = nv_model.true_eval(saa_sol) - nv_model.CV_performance(data, fold_val, 'saa')

            # for k, btsp_val in enumerate(BOOTSTRAP_LIST):
            #     error[choice_num * j + 4 + len(KFOLD_LIST) + k][i] = nv_model.true_eval(saa_sol) - 2 * nv_model.empirical_eval(saa_sol, data) + nv_model.bootstrap_debias(data, btsp_val, 'saa')[0]

            # error[choice_num * (j + 1) - 1][i] = nv_model.true_eval(saa_sol) - sample_num * nv_model.empirical_eval(saa_sol, data) + (sample_num - 1) * nv_model.jackknife_debias(data, 'saa')[0]

            normal_sol = nv_model.normal(data)
            j = 1
            error[choice_num * j][i] = nv_model.true_eval(normal_sol)
            error[choice_num* j + 1][i] = nv_model.true_eval(normal_sol) - nv_model.empirical_eval(normal_sol, data)
            error[choice_num* j + 2][i] = error[choice_num* j + 1][i] - nv_model.normal_correction(normal_sol, data)
            error[choice_num* j + 3][i] = error[choice_num* j][i] - (nv_model.normal_eval(normal_sol, data) + nv_model.normal_correction_MS(normal_sol, data))
            # for k, fold_val in enumerate(KFOLD_LIST):
            #     error[choice_num * j + 4 + k][i] = nv_model.true_eval(normal_sol) - nv_model.CV_performance(data, fold_val, 'normal')
            # # for k, btsp_val in enumerate(BOOTSTRAP_LIST):
            #     error[choice_num * j + 3 + len(KFOLD_LIST) + k][i] = nv_model.true_eval(normal_sol) - 2 * nv_model.empirical_eval(normal_sol, data) + nv_model.bootstrap_debias(data, btsp_val, 'normal')[0]

            # error[choice_num * (j + 1) - 1][i] = nv_model.true_eval(normal_sol) - sample_num * nv_model.empirical_eval(normal_sol, data) + (sample_num - 1) * nv_model.jackknife_debias(data, 'normal')[0]


            exp_sol = nv_model.exp(data)
            j = 2
            error[choice_num * j][i] = nv_model.true_eval(exp_sol)
            error[choice_num* j + 1][i] = nv_model.true_eval(exp_sol) - nv_model.empirical_eval(exp_sol, data)
            error[choice_num* j + 2][i] = error[choice_num* j + 1][i] - nv_model.exp_correction(exp_sol, data)
            
            # nv_model.approx = True
            # error[choice_num* j + 2][i] = error[choice_num* j][i] - nv_model.exp_correction(exp_sol, data)
            error[choice_num* j + 3][i] = error[choice_num* j][i] - (nv_model.exp_eval(exp_sol, data) + nv_model.exp_correction_MS(exp_sol, data))
            # for k, fold_val in enumerate(KFOLD_LIST):
            #     error[choice_num * j + 4 + k][i] = nv_model.true_eval(exp_sol) - nv_model.CV_performance(data, fold_val, 'exponential')
            # for k, btsp_val in enumerate(BOOTSTRAP_LIST):
            #     error[choice_num * j + 4 + len(KFOLD_LIST) + k][i] = nv_model.true_eval(exp_sol) - 2 * nv_model.empirical_eval(exp_sol, data) + nv_model.bootstrap_debias(data, btsp_val, 'exponential')[0]

            # error[choice_num * (j + 1) - 1][i] = nv_model.true_eval(exp_sol) - sample_num * nv_model.empirical_eval(exp_sol, data) + (sample_num - 1) * nv_model.jackknife_debias(data, 'exponential')[0]


            exp_OS_sol = nv_model.OS_exp(data)
            j = 3
            error[choice_num * j][i] = nv_model.true_eval(exp_OS_sol)
            error[choice_num* j + 1][i] = nv_model.true_eval(exp_OS_sol) - nv_model.empirical_eval(exp_OS_sol, data)
            error[choice_num* j + 2][i] = error[choice_num* j + 1][i] - nv_model.OS_exp_correction(exp_OS_sol, data)
            error[choice_num* j + 3][i] = error[choice_num* j][i] -(nv_model.exp_eval(exp_OS_sol, data) + nv_model.exp_correction_MS(exp_OS_sol, data))
            # for k, fold_val in enumerate(KFOLD_LIST):
            #     error[choice_num * j + 4 + k][i] = nv_model.true_eval(exp_OS_sol) - nv_model.CV_performance(data, fold_val, 'exp_OS')
            # for k, btsp_val in enumerate(BOOTSTRAP_LIST):
            #     error[choice_num * j + 3 + len(KFOLD_LIST) + k][i] = nv_model.true_eval(exp_OS_sol) - 2 * nv_model.empirical_eval(exp_OS_sol, data) + nv_model.bootstrap_debias(data, btsp_val, 'exp_OS')[0]

            # error[choice_num * (j + 1) - 1][i] = nv_model.true_eval(exp_OS_sol) - sample_num * nv_model.empirical_eval(exp_OS_sol, data) + (sample_num - 1) * nv_model.jackknife_debias(data, 'exp_OS')[0]

        #for i in range(loop_num):
        error_avg[s] = np.mean(error, axis = 1)


    import pandas as pd

    df = pd.DataFrame(error_avg)
    df.to_csv(f'output/nv/nv_evaluate2_{sample_num}_{mode}.csv')


if __name__ == '__main__':
    #run(25, 'normal')
    run(50, 'normal')
    # run(75, 'normal')
    #run(100, 'normal')
    # run(125, 'normal')
    #run(25, 'exponential')
    run(50, 'exponential')
    # run(75, 'exponential')
    #run(100, 'exponential')
    # run(125, 'exponential')
