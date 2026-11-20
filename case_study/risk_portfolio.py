# mean-variance portfolio allocation
from pathlib import Path
import sys

# Allow direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import time


from src.portfolio_model import mv_portfolio
from src.paths import output_path

#sample_num = 20
# base case: all choose the same weigght

v = 50
def risk_func(x):
    return max(0, -(1 - x) ** 2 + v)

def risk_func_grad(x):
    return 2 * (1 - x)
def risk_func_hessian(x):
    return 2

def base_case(sample_num, asset_num):

    loop_num = 50
    outer_num = 20

    eval_num = 7
 

    dro_coefs = [3]
    kfold_num = [5, 10]
    choice_num = 4 + len(dro_coefs)
    eval_num = 4 + len(kfold_num)

    model_perform = np.zeros((loop_num, choice_num * eval_num))
    time_num = 2 + len(kfold_num)
    time_p = np.zeros((loop_num, choice_num  * (2 + len(kfold_num))))

    model_avg = np.zeros((outer_num, choice_num * eval_num))
    time_avg = np.zeros((outer_num, choice_num * (2 + len(kfold_num))))



    assert (asset_num % 2 == 0)

    all_eval = True
    for s in range(outer_num):
        print(s)
        # DGP
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

        # for i in range(asset_num):
        #     for j in range(i + 1, asset_num):
        #         rad_val = 0.5 * np.random.rand() 
        #         return_cov[i][j], return_cov[j][i] = 0.5, 0.5
        for k in range(loop_num):
            print(s, k)


            lambda_1 = 1
            lambda_2 = 2
            assert (lambda_1 >=0 and lambda_2 >= 0)


            mv_model = mv_portfolio(return_mean, cov_all, lambda_1, lambda_2)

            eval_true = np.zeros((loop_num))


            data = mv_model.generate(sample_num)

            j = 0
            saa_sol = mv_model.saa_sol(data)
            model_perform[k][eval_num * j + 0] = risk_func(mv_model.true_eval(saa_sol))
            center_pt = mv_model.sample_eval(saa_sol, data)
            model_perform[k][eval_num * j + 1] = risk_func(center_pt)
            model_perform[k][eval_num * j + 2] = risk_func(center_pt) + mv_model.saa_correction(saa_sol, data) * risk_func_grad(center_pt) - risk_func_hessian(center_pt) / 2 * mv_model.sample_obj_var(saa_sol, data)

            #time_p[k][time_num * j + 0] = time.time() - start_time

            # if all_eval == True:

            #     for kn, K in enumerate(kfold_num):
            #         start_time = time.time()
            #         model_perform[k][eval_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'saa')
            #         time_p[k][time_num * j + 1 + kn] = time.time() - start_time

                
            #     start_time = time.time()
            #     #model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'saa')
            #     model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.saa_alo(saa_sol, data)
            #     time_p[k][time_num * j + 3] = time.time() - start_time

            # print('==============')

            j = 1
            uniform_sol = mv_model.uniform_sol(data)
            model_perform[k][eval_num * j + 0] = risk_func(mv_model.true_eval(uniform_sol))
            center_pt = mv_model.sample_eval(uniform_sol, data)
            model_perform[k][eval_num * j + 1] = risk_func(center_pt)

            start_time = time.time()
            model_perform[k][eval_num * j + 2] = risk_func(center_pt) + mv_model.uniform_correction(uniform_sol, data) * risk_func_grad(center_pt) - risk_func_hessian(center_pt) / 2 * mv_model.sample_obj_var(uniform_sol, data)
            # time_p[k][time_num * j + 0] = time.time() - start_time



            # if all_eval == True:
                
            #     for kn, K in enumerate(kfold_num):
            #         start_time = time.time()
            #         model_perform[k][eval_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'uniform')
            #         time_p[k][time_num * j + 1 + kn] = time.time() - start_time


            #     start_time = time.time()
            #     #model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'uniform')
            #     model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.uniform_saa_alo(uniform_sol, data)
            #     time_p[k][time_num * j + 3] = time.time() - start_time
            # print('==============')                


            j = 2
            block_sol = mv_model.block_sol(data)
            model_perform[k][eval_num * j + 0] = risk_func(mv_model.true_eval(block_sol))
            center_pt = mv_model.sample_eval(block_sol, data)
            model_perform[k][eval_num * j + 1] = risk_func(center_pt)

            start_time = time.time()
            model_perform[k][eval_num * j + 2] = risk_func(center_pt) + mv_model.block_correction(block_sol, data) * risk_func_grad(center_pt) - risk_func_hessian(center_pt) / 2 * mv_model.sample_obj_var(block_sol, data)
            # time_p[k][time_num * j + 0] = time.time() - start_time

            # if all_eval == True:
                
            #     for kn, K in enumerate(kfold_num):
            #         start_time = time.time()
            #         model_perform[k][eval_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'block')
            #         time_p[k][time_num * j + 1 + kn] = time.time() - start_time



            #     #model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'block')
            #     start_time = time.time()
            #     model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.block_saa_alo(block_sol, data)
            #     time_p[k][time_num * j + 3] = time.time() - start_time
            # print('===============')

            j = 3
            param_sol = mv_model.normal_idpd_opt(data)
            model_perform[k][eval_num * j + 0] = risk_func(mv_model.true_eval(param_sol))

            center_pt = mv_model.sample_eval(param_sol, data)
            model_perform[k][eval_num * j + 1] = risk_func(center_pt)
            
            # start_time = time.time()
            model_perform[k][eval_num * j + 2] =  risk_func(center_pt) + mv_model.normal_idpd_pf_correction(param_sol, data) * risk_func_grad(center_pt) - risk_func_hessian(center_pt) / 2 * mv_model.sample_obj_var(param_sol, data)       
            # time_p[k][time_num * j + 0] = time.time() - start_time

            # if all_eval == True:
            #     start_time = time.time()
            #     for kn, K in enumerate(kfold_num):
            #         model_perform[k][eval_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'normal')
            #         time_p[k][time_num * j + 1 + kn] = time.time() - start_time

            #     start_time = time.time()
            #     #model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'normal')
            #     model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.normal_idpd_alo(param_sol, data)
            #     time_p[k][time_num * j + 3] = time.time() - start_time

            # print('======================')
            # # DRO solutions
            for i, coef in enumerate(dro_coefs):
                mv_model.ambiguity_coef = coef
                j = 4 + i
                dro_sol = mv_model.chi2_DRO_sol(data)
            
                model_perform[k][eval_num * j + 0] = risk_func(mv_model.true_eval(dro_sol))
                center_pt = mv_model.sample_eval(dro_sol, data)
                model_perform[k][eval_num * j + 1] = risk_func(center_pt)
                model_perform[k][eval_num * j + 2] = risk_func(center_pt) + mv_model.saa_correction(dro_sol, data) * risk_func_grad(center_pt) - risk_func_hessian(center_pt) / 2 * mv_model.sample_obj_var(dro_sol, data)    
            #     time_p[k][time_num * j + 0] = (time.time() - start_time)

            #     if all_eval == True:
                    
            #         for kn, K in enumerate(kfold_num):
            #             start_time = time.time()
            #             model_perform[k][eval_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'chi2dro')
            #             time_p[k][time_num * j + kn + 1] = time.time() - start_time

            #         #model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'chi2dro')
            #         start_time = time.time()
            #         model_perform[k][eval_num * j + 3 + len(kfold_num)] = mv_model.saa_alo(dro_sol, data)
            #         time_p[k][time_num * j + 3] = time.time() - start_time





        model_avg[s] = np.mean(model_perform, axis = 0)
        time_avg[s] = np.mean(time_p, axis = 0)
    



    import pandas as pd
    #print(np.mean(time_avg, axis = 0))

    df = pd.DataFrame(model_avg)
    # set2 = df[17] - df[1]
    # print(np.mean(set2), np.std(set2))
    df.to_csv(output_path('portfolio', f'portfolio_risk_{asset_num}_{sample_num}_test.csv'))

    print(np.mean(model_avg, axis = 0))



if __name__ == '__main__':
    base_case(50, 10)
    # base_case(50, 20)
    # base_case(100, 10)
    # base_case(100, 20)



# compare average model performance here




# selection criterion




