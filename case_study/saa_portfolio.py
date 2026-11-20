# mean-variance portfolio allocation
from pathlib import Path
import sys

# Allow direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import math


from src.portfolio_model import mv_portfolio
from src.paths import figure_path

import matplotlib.pyplot as plt
import seaborn as sns

#sample_num = 20
# base case: a

loop_num = 50
outer_num = 50



asset_nums = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
#asset_nums = [6, 12]

eval_num = 1
choice_num = 3
lambda_1 = 1
lambda_2 = 2


# dro_coefs = [1, 3, 5]
# kfold_num = [2, 5, 10]

# TODO: vary sample size to be 
sample_num = 50

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


def main():
    """Run the portfolio comparison and save its figure."""
    model_perform = np.zeros((outer_num, choice_num * eval_num))

    final_perform_mean = np.zeros((2, len(asset_nums), eval_num * choice_num))
    final_perform_std = np.zeros((2, len(asset_nums), eval_num * choice_num))

    labels = ['Oracle', 'OIC', '5-CV']

    sns.set_style('darkgrid')

    plt.figure()
    linestyles = ['-', '--']

    for j, sample_num in enumerate([50, 100]):
        for i, asset_num in enumerate(asset_nums):
            for s in range(outer_num):
                print(sample_num, asset_num, s)
                model_inner = np.zeros((loop_num, choice_num * eval_num))
                for k in range(loop_num):
                    return_mean, cov_all = DGP(asset_num)
                    mv_model = mv_portfolio(return_mean, cov_all, lambda_1, lambda_2)
                    data = mv_model.generate(sample_num)

                    saa_sol = mv_model.saa_sol(data)
                    model_inner[k][0] = mv_model.true_eval(saa_sol) - mv_model.sample_eval(saa_sol, data)
                    model_inner[k][1] = mv_model.saa_correction(saa_sol, data)
                    model_inner[k][2] = mv_model.CV_performance(data, 5, 'saa') - mv_model.sample_eval(saa_sol, data)
                
                model_perform[s] = np.mean(model_inner, axis = 0)


            final_perform_mean[j][i] = np.mean(model_perform, axis = 0)
            final_perform_std[j][i] = np.std(model_perform, axis = 0)




        final_mean = final_perform_mean[j].T
        final_std = final_perform_std[j].T

        marker_choice = ['*', 'o', 'v', '1', '2', '3']
        colors=['#4daf4a', '#ff7f00', '#984ea3', '#FFD43B', '#a65628', '#f781bf', '#e41a1c', '#377eb8']

        for k in range(len(labels)):
            plt.plot(asset_nums, final_mean[k], label = labels[k] + f', n = {sample_num}', linestyle = linestyles[j], marker = marker_choice[k], color = colors[k], linewidth = 1)
            plt.errorbar(x = asset_nums, y = final_mean[k], yerr = final_std[k]/math.sqrt(outer_num), color = colors[k], capsize = 2)
    plt.legend(loc = 'upper right')
    plt.xlabel('Asset Nums')
    plt.ylabel('Decision Complexity Value')

    plt.savefig(figure_path('portfolio', 'test2.pdf'))


if __name__ == '__main__':
    main()
