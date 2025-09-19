#!/user/tw2837/.conda/envs/tw2837/bin/python
import src.datasets.tabular
import src.datasets.utils
from src.datasets.tabular import get_dataset_config
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import time

from src.models import *
from src.model_utils import * 
set_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


eta = 0.3
fold_num = 5 # fold number to use

poly_size = 2
alpha0 = 1
model_class = 'ridge'
CV_compare = True # include cross validation or not
train_frac = 0.1

dataset = 'wine'

# approximate leave-one-out for each method
def debias_wine_AOCV():
    time1, time2 = 0, 0
    randint = np.random.randint(0, 1000, size = 1)[0]
    dataset_kwargs = {'root_dir': 'src/datasets', 'random_state': randint, 'is_regression': True, 'train_frac': train_frac}

    dataset_config = get_dataset_config(dataset, **dataset_kwargs)

    loop_times = int((1 - train_frac) / train_frac)
    dset = src.datasets.utils.get_dataset(dataset_config)
    X_tr, y_tr, X_te, y_te, X_val, y_val = dset.get_data()

    X_te = pd.concat([X_te, X_val])
    y_te = pd.concat([y_te, y_val])

    if poly_size > 1:
        poly = PolynomialFeatures(poly_size)
        X_tr = poly.fit_transform(X_tr)
        X_te = poly.fit_transform(X_te)

    else:
        if model_class != 'nn':
            X_tr['intercept'] = 1
            for i in range(loop_times):
                dset.X_te_list[i]['intercept'] = 1    
        X_tr = X_tr.to_numpy()
    y_tr = y_tr.to_numpy()

    if model_class == 'ridge':
        model = Ridge(fit_intercept = False, alpha = alpha0)
    elif model_class == 'linear':      
        model = LinearRegression(fit_intercept = False)
    elif model_class == 'nn':
        model = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
        config = {"lr": 0.1, "batch_size": 128, "train_epochs": 500, "hidden_size": 8, "dropout_ratio": 0.1}
        model.update(config)
    elif model_class == 'nn2':
        model = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
        config = {"lr": 0.2, "batch_size": 128, "train_epochs": 500, "hidden_size": 16, "dropout_ratio": 0.1}
        model.update(config)

    model.fit(X_tr, y_tr)
    # OIC DEBIASING
    y_tr_pred = model.predict(X_tr)
    residual = y_tr - y_tr_pred
    loss = (y_tr_pred - y_tr) ** 2
    in_sample = np.mean(np.maximum(loss - eta, 0))

    print('in_sample result', in_sample)

    #OOS result
    out_sample = np.zeros(loop_times)
    for i in range(loop_times):
        X_te = dset.X_te_list[i].values
        y_te = dset.y_te_list[i].values
        if poly_size > 1:
            X_te = poly.fit_transform(X_te)
        y_te_pred = model.predict(X_te)
        loss = (y_te_pred - y_te) ** 2

        out_sample[i] = np.mean(np.maximum(loss - eta, 0))
    print('OOS result', np.mean(out_sample))

    # OIC DEBIASING

    OIC_debias = 0
    start_time = time.time()
    while True:
        grad_interact_sum = torch.zeros(1).to(set_device)
        if model_class not in ['nn', 'nn2']:
            if model_class == 'ridge':
                cov_inv = np.linalg.inv(np.dot(X_tr.T, X_tr)/len(X_tr) + (alpha0 / len(X_tr) * np.identity(len(X_tr[0]))))
            else:
                cov_inv = np.linalg.pinv(np.dot(X_tr.T, X_tr)/len(X_tr))

            for i in range(len(X_tr)):
                if loss[i] > eta:
                    grad_interact = 2 * residual[i] * X_tr[i]
                    grad_interact_sum += -poly_size * np.dot(cov_inv, X_tr[i] * residual[i]).dot(grad_interact) / (len(X_tr) ** 2)


        else:
            #obtain gradient from nn
            inp_X = torch.from_numpy(X_tr)
            inp_y = torch.from_numpy(y_tr)
            inp_X.to(set_device)
            inp_y.to(set_device)
            gradients, empirical_hessian, __ = nn_gradient_hessian(model.model, inp_X, inp_y)
            cov_inv = torch.inverse(empirical_hessian.cpu()).to(set_device)
            print(cov_inv.shape)
            for i in range(len(X_tr)):
                if loss[i] > eta:

                    grad_interact_sum += -2 * torch.mm(gradients[i].reshape(1, -1), cov_inv).mm(gradients[i].reshape(-1, 1))[0] / (len(X_tr) ** 2)
        OIC_debias = torch.Tensor([in_sample]).to(set_device) - grad_interact_sum

        if model_class not in ['nn', 'nn2'] or (model_class in ['nn', 'nn2'] and (OIC_debias > torch.Tensor([in_sample]).to(set_device) and OIC_debias < 1)):
            break
        # we fit again
        model = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
        if model_class == 'nn':
            config = {"lr": 0.1, "batch_size": 128, "train_epochs": 500, "hidden_size": 8, "dropout_ratio": 0.1}
        else:
            config = {"lr": 0.2, "batch_size": 128, "train_epochs": 500, "hidden_size": 16, "dropout_ratio": 0.1}
        model.update(config)
        model.fit(X_tr, y_tr)
    time1 = time.time() - start_time
    debias_result = 0

    if CV_compare == True:
        start_time = time.time()
        # CV
        kf = KFold(n_splits = fold_num, shuffle = True)
        cv_sample = []
        for train_label, test_label in kf.split(X_tr):
            X_tr1 = X_tr[train_label]
            y_tr1 = y_tr[train_label]

            X_te1 = X_tr[test_label]
            y_te1 = y_tr[test_label]
            if model_class == 'ridge':
                model1 = Ridge(fit_intercept = False, alpha = alpha0)
            elif model_class == 'linear':
                model1 = LinearRegression(fit_intercept = False)
            elif model_class == 'nn':
                model1 = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
                config = {"lr": 0.1, "batch_size": 128, "train_epochs": 500, "hidden_size": 8, "dropout_ratio": 0.1}
                model1.update(config)
            elif model_class == 'nn2':
                model1 = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
                config = {"lr": 0.2, "batch_size": 128, "train_epochs": 500, "hidden_size": 16, "dropout_ratio": 0.1}
                model1.update(config)

            model1.fit(X_tr1, y_tr1)
            y_te1_pred = model1.predict(X_te1)
            loss1 = (y_te1_pred - y_te1) ** 2
            cv_sample.append(np.mean(np.maximum(loss1 - eta, 0)))
        print('CV result', np.mean(cv_sample))
        print('CV difference', np.mean(out_sample) - np.mean(cv_sample))
        time3 = time.time() - start_time

    start_time = time.time()

    # Due to instability of hessian and gradient computation in NNs, 
    # we only return the bias when the bias sign is positive.
    while True:
        single_loss = np.zeros(len(X_tr))
        if model_class in ['nn', 'nn2']:
            inp_X = torch.from_numpy(X_tr)
            inp_y = torch.from_numpy(y_tr)
            inp_X.to(set_device)
            inp_y.to(set_device)
            gradients, empirical_hessian, node_nums = nn_gradient_hessian(model.model, inp_X, inp_y)
            cov_inv = torch.inverse(empirical_hessian.cpu()).to(set_device)
            for i in range(len(X_tr)):
                update_val = torch.mm(cov_inv, gradients[i].reshape(-1, 1)) / len(X_tr)
                cum = 0
                for k, p in enumerate(model.model.parameters()):
                    p.data += update_val[cum:(cum + node_nums[k])].reshape(p.data.shape)
                    cum += node_nums[k]

                y_tr_pred = model.predict(X_tr[i:(i + 1)])
                single_loss[i] = max((y_tr_pred[0] - y_tr[i]) ** 2 - eta, 0)
                #recover the default value
                cum = 0
                for k, p in enumerate(model.model.parameters()):
                    p.data -= update_val[cum:(cum + node_nums[k])].reshape(p.data.shape)
                    cum += node_nums[k]


        else:
            # AOCV
            theta_default = model.coef_

            new_X = X_tr
            if model_class == 'ridge':
                cov_inv = np.linalg.inv(np.dot(new_X.T, new_X)/(len(X_tr) - 1) + (alpha0 / (len(X_tr) - 1) * np.identity(len(X_tr[0]))))
            else:
                cov_inv = np.linalg.pinv(np.dot(new_X.T, new_X)/(len(X_tr) - 1))
            #remove the i-th sample
            for i in range(len(X_tr)):
                #print(np.dot(cov_inv, X_tr[i] * residual[i]))
                theta_acv = theta_default - (np.dot(cov_inv, X_tr[i] * residual[i]))/(len(X_tr) - 1)
                # set new model parameter
                model.coef_ = theta_acv
                y_tr_pred2 = model.predict(X_tr[i:(i + 1)])
                #print(theta_default, theta_acv)
                single_loss[i] = max((y_tr[i] - y_tr_pred2[0]) ** 2 - eta, 0)
            
        debias_result = np.mean(single_loss)
        print('ALO Debiasing Performance', debias_result)

        if model_class not in ['nn', 'nn2'] or (model_class in ['nn', 'nn2'] and (debias_result > torch.Tensor([in_sample]).to(set_device) and debias_result < 1)):
            break
        # we fit again
        model = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
        if model_class == 'nn':
            config = {"lr": 0.1, "batch_size": 128, "train_epochs": 500, "hidden_size": 8, "dropout_ratio": 0.1}
        else:
            config = {"lr": 0.2, "batch_size": 128, "train_epochs": 500, "hidden_size": 16, "dropout_ratio": 0.1}
        model.update(config)
        model.fit(X_tr, y_tr)
    time2 = time.time() - start_time

    #return np.mean(single_loss)
    print(time1, time2, time3)
    return in_sample, np.mean(out_sample), debias_result, OIC_debias, time1, time2, time3


# OIC of each method
def debias_wine():

    randint = np.random.randint(0, 1000, size = 1)[0]
    dataset_kwargs = {'root_dir': 'src/datasets', 'random_state': randint, 'is_regression': True, 'train_frac': train_frac}

    dataset_config = get_dataset_config(dataset, **dataset_kwargs)

    loop_times = int((1 - train_frac) / train_frac)
    dset = src.datasets.utils.get_dataset(dataset_config)
    X_tr, y_tr, X_te, y_te, X_val, y_val = dset.get_data()

    X_te = pd.concat([X_te, X_val])
    y_te = pd.concat([y_te, y_val])

    if poly_size > 1:
        poly = PolynomialFeatures(poly_size)
        X_tr = poly.fit_transform(X_tr)
        X_te = poly.fit_transform(X_te)

    else:
        if model_class != 'nn':
            X_tr['intercept'] = 1
            for i in range(loop_times):
                dset.X_te_list[i]['intercept'] = 1    
        X_tr = X_tr.to_numpy()
    y_tr = y_tr.to_numpy()

    if model_class == 'ridge':
        model = Ridge(fit_intercept = False, alpha = alpha0)
    elif model_class == 'linear':      
        model = LinearRegression(fit_intercept = False)
    elif model_class == 'nn':
        model = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
        config = {"lr": 0.1, "batch_size": 128, "train_epochs": 500, "hidden_size": 8, "dropout_ratio": 0.1}
        model.update(config)
    elif model_class == 'nn2':
        model = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
        config = {"lr": 0.2, "batch_size": 128, "train_epochs": 500, "hidden_size": 12, "dropout_ratio": 0.1}
        model.update(config)

    model.fit(X_tr, y_tr)



    y_tr_pred = model.predict(X_tr)
    residual = y_tr - y_tr_pred
    loss = (y_tr_pred - y_tr) ** 2
    in_sample = np.mean(np.maximum(loss - eta, 0))


    # Due to instability of hessian and gradient computation in NNs, 
    # we only return the bias when the bias sign is positive.
    while True:
        grad_interact_sum = torch.zeros(1).to(set_device)
        if model_class not in ['nn', 'nn2']:
            if model_class == 'ridge':
                cov_inv = np.linalg.inv(np.dot(X_tr.T, X_tr)/len(X_tr) + (alpha0 / len(X_tr) * np.identity(len(X_tr[0]))))
            else:
                cov_inv = np.linalg.pinv(np.dot(X_tr.T, X_tr)/len(X_tr))

            for i in range(len(X_tr)):
                if loss[i] > eta:
                    grad_interact = 2 * residual[i] * X_tr[i]
                    grad_interact_sum += -poly_size * np.dot(cov_inv, X_tr[i] * residual[i]).dot(grad_interact) / (len(X_tr) ** 2)


        else:
            #obtain gradient from nn
            inp_X = torch.from_numpy(X_tr)
            inp_y = torch.from_numpy(y_tr)
            inp_X.to(set_device)
            inp_y.to(set_device)
            gradients, empirical_hessian = nn_gradient_hessian(model.model, inp_X, inp_y)
            cov_inv = torch.inverse(empirical_hessian.cpu()).to(set_device)
            print(cov_inv.shape)
            for i in range(len(X_tr)):
                if loss[i] > eta:

                    grad_interact_sum += -2 * torch.mm(gradients[i].reshape(1, -1), cov_inv).mm(gradients[i].reshape(-1, 1))[0] / (len(X_tr) ** 2)
        debias = torch.Tensor([in_sample]).to(set_device) - grad_interact_sum

        if model_class not in ['nn', 'nn2'] or (model_class in ['nn', 'nn2'] and (debias > torch.Tensor([in_sample]).to(set_device) and debias < 1)):
            break
        # we fit again
        model = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
        if model_class == 'nn':
            config = {"lr": 0.1, "batch_size": 128, "train_epochs": 500, "hidden_size": 8, "dropout_ratio": 0.1}
        else:
            config = {"lr": 0.2, "batch_size": 128, "train_epochs": 500, "hidden_size": 16, "dropout_ratio": 0.1}
        model.update(config)

        model.fit(X_tr, y_tr)



    print('debiasing result', debias)

    #OOS result
    out_sample = np.zeros(loop_times)
    for i in range(loop_times):
        X_te = dset.X_te_list[i].values
        y_te = dset.y_te_list[i].values
        if poly_size > 1:
            X_te = poly.fit_transform(X_te)
        y_te_pred = model.predict(X_te)
        loss = (y_te_pred - y_te) ** 2

        out_sample[i] = np.mean(np.maximum(loss - eta, 0))
    print('OOS result', np.mean(out_sample))
    print('debiasing difference', np.mean(out_sample) - in_sample + grad_interact_sum)
    print('==========')
    if CV_compare == True:
        # CV
        kf = KFold(n_splits = fold_num, shuffle = True)
        cv_sample = []
        for train_label, test_label in kf.split(X_tr):
            X_tr1 = X_tr[train_label]
            y_tr1 = y_tr[train_label]

            X_te1 = X_tr[test_label]
            y_te1 = y_tr[test_label]
            if model_class == 'ridge':
                model1 = Ridge(fit_intercept = False, alpha = alpha0)
            elif model_class == 'linear':
                model1 = LinearRegression(fit_intercept = False)
            elif model_class == 'nn':
                model1 = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
                config = {"lr": 0.1, "batch_size": 128, "train_epochs": 500, "hidden_size": 8, "dropout_ratio": 0.1}
                model1.update(config)
            elif model_class == 'nn2':
                model1 = MLPRegressor(input_dim=len(X_tr[0]), num_classes = 1)
                config = {"lr": 0.2, "batch_size": 128, "train_epochs": 500, "hidden_size": 16, "dropout_ratio": 0.1}
                model1.update(config)

            model1.fit(X_tr1, y_tr1)
            y_te1_pred = model1.predict(X_te1)
            loss1 = (y_te1_pred - y_te1) ** 2
            cv_sample.append(np.mean(np.maximum(loss1 - eta, 0)))
        print('CV result', np.mean(cv_sample))
        print('CV difference', np.mean(out_sample) - np.mean(cv_sample))
        
        return in_sample, debias.cpu().numpy()[0], np.mean(cv_sample), np.mean(out_sample)
    else:
        return in_sample, debias.cpu().numpy()[0], 0, np.mean(out_sample)

if __name__ == '__main__':
    loop_num = 10
    result = np.zeros((loop_num, 4))
    time1_s = np.zeros(loop_num)
    time2_s = np.zeros(loop_num)
    time3_s = np.zeros(loop_num)
    for i in range(loop_num):
        result[i][0], result[i][1], result[i][2], result[i][3], time1_s[i], time2_s[i], time3_s[i] = debias_wine_AOCV()
    #print(result)
    print('avg', np.mean(result, axis = 0), np.std(result, axis = 0))
    print('avg time', np.mean(time1_s), np.mean(time2_s), np.mean(time3_s))
