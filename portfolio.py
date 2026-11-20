# mean-variance portfolio allocation
import numpy as np


from src.portfolio_model import mv_portfolio
from src.paths import output_path

#sample_num = 20
# base case: all choose the same weigght

def base_case(sample_num, asset_num):

    loop_num = 50
    outer_num = 20

    eval_num = 7
    choice_num = 3 + 4

    dro_coefs = [1, 3, 5]
    kfold_num = [2, 5, 10]

    model_perform = np.zeros((loop_num, choice_num * eval_num))

    model_avg = np.zeros((outer_num, choice_num * eval_num))



    assert (asset_num % 2 == 0)

    all_eval = True
    for s in range(outer_num):

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
            model_perform[k][choice_num * j + 0] = mv_model.true_eval(saa_sol)
            model_perform[k][choice_num * j + 1] = mv_model.sample_eval(saa_sol, data)
            model_perform[k][choice_num * j + 2] = model_perform[k][choice_num * j + 1] + mv_model.saa_correction(saa_sol, data)

            if all_eval == True:

                for kn, K in enumerate(kfold_num):
                    model_perform[k][choice_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'saa')

                model_perform[k][choice_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'saa')

            j = 1
            uniform_sol = mv_model.uniform_sol(data)
            model_perform[k][choice_num * j + 0] = mv_model.true_eval(uniform_sol)
            model_perform[k][choice_num * j + 1] = mv_model.sample_eval(uniform_sol, data)
            model_perform[k][choice_num * j + 2] = model_perform[k][choice_num * j + 1] + mv_model.uniform_correction(uniform_sol, data)
            
            if all_eval == True:
                for kn, K in enumerate(kfold_num):
                    model_perform[k][choice_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'uniform')

                model_perform[k][choice_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'uniform')



            j = 2
            block_sol = mv_model.block_sol(data)
            model_perform[k][choice_num * j + 0] = mv_model.true_eval(block_sol)
            model_perform[k][choice_num * j + 1] = mv_model.sample_eval(block_sol, data)
            model_perform[k][choice_num * j + 2] = model_perform[k][choice_num * j + 1] + mv_model.block_correction_torch(block_sol, data)

            if all_eval == True:
                for kn, K in enumerate(kfold_num):
                    model_perform[k][choice_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'block')

                model_perform[k][choice_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'block')


            j = 3
            param_sol = mv_model.normal_idpd_opt(data)
            model_perform[k][choice_num * j + 0] = mv_model.true_eval(param_sol)
            model_perform[k][choice_num * j + 1] = mv_model.sample_eval(param_sol, data)
            model_perform[k][choice_num * j + 2] = model_perform[k][choice_num * j + 1] + mv_model.normal_idpd_pf_correction(param_sol, data)

            if all_eval == True:
                for kn, K in enumerate(kfold_num):
                    model_perform[k][choice_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'normal')

                model_perform[k][choice_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'normal')


            # DRO solutions
            for i, coef in enumerate(dro_coefs):
                mv_model.ambiguity_coef = coef
                j = 4 + i
                dro_sol = mv_model.chi2_DRO_sol(data)
            
                model_perform[k][choice_num * j + 0] = mv_model.true_eval(dro_sol)
                model_perform[k][choice_num * j + 1] = mv_model.sample_eval(dro_sol, data)
                model_perform[k][choice_num * j + 2] = model_perform[k][choice_num * j + 1] + mv_model.saa_correction(dro_sol, data)

                if all_eval == True:
                    for kn, K in enumerate(kfold_num):
                        model_perform[k][choice_num * j + 3 + kn] = mv_model.CV_performance(data, K, 'chi2dro')

                    model_perform[k][choice_num * j + 3 + len(kfold_num)] = mv_model.bootstrap_debias(data, 10, 'chi2dro')




        model_avg[s] = np.mean(model_perform, axis = 0)



    import pandas as pd

    df = pd.DataFrame(model_avg)
    df.to_csv(output_path('portfolio', f'portfolio_{asset_num}_{sample_num}_test.csv'))





if __name__ == '__main__':
    base_case(50, 20)
    base_case(100, 10)
    base_case(100, 20)



# compare average model performance here




# selection criterion




