#!/usr/bin/env python3
"""Wine regression: OIC, IFC-OIC, CV, and infinitesimal-jackknife ALO.

Adapted from oic_codes/ml_task_new.py, with the neural-network architecture
and training settings from oic_codes/ml_task.py.
"""
import numpy as np
import torch
from torch.func import functional_call
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

import src.datasets.utils
from src.datasets.tabular import get_dataset_config
from src.models import MLPRegressor
from src.paths import DATA_ROOT


poly_size = 2
alpha0 = 5
model_class = 'ridge'  # 'linear', 'ridge', or 'nn'
CV_compare = True
fold_num = 5
train_frac = 0.3
loop_num = 100

dataset = 'wine'
eta = 0.3

# Same NN architecture and optimizer settings as ml_task.py. alpha0 is used
# only for ridge; the original neural-network fit is unregularized.
nn_config = {'lr': 0.1, 'batch_size': 128, 'train_epochs': 500,
             'hidden_size': 8, 'dropout_ratio': 0.1}
nn_curvature_batch_size = 128
# The existing MLP evaluates in float32; discard curvature below its accuracy.
nn_pinv_rtol = 1e-6

RESULT_NAMES = ('EM', 'OIC', 'IFC-OIC', 'CV', 'ALO-IJ',
                'First-order LOO', 'LOO', 'GCV', 'OOS')


def get_train_test_data(seed):
    dataset_kwargs = {
        'root_dir': str(DATA_ROOT),
        'random_state': seed, 'is_regression': True, 'train_frac': train_frac,
    }
    dataset_config = get_dataset_config(dataset, **dataset_kwargs)
    dset = src.datasets.utils.get_dataset(dataset_config)
    X_tr, y_tr, X_te, y_te, X_val, y_val = dset.get_data()

    if poly_size > 1:
        poly = PolynomialFeatures(poly_size)
        X_tr = poly.fit_transform(X_tr)
        X_te_list = [poly.transform(X_te_split) for X_te_split in dset.X_te_list]
    elif model_class == 'nn':
        X_tr = X_tr.to_numpy()
        X_te_list = [split.to_numpy() for split in dset.X_te_list]
    else:
        X_tr = np.column_stack([np.ones(len(X_tr)), X_tr.to_numpy()])
        X_te_list = [np.column_stack([np.ones(len(split)), split.to_numpy()])
                     for split in dset.X_te_list]

    y_tr = y_tr.to_numpy()
    y_te_list = [y_te_split.values for y_te_split in dset.y_te_list]
    return X_tr, y_tr, X_te_list, y_te_list


def make_nn(input_dim, seed):
    torch.manual_seed(seed)
    model = MLPRegressor(input_dim=input_dim, num_classes=1)
    model.update(nn_config)
    return model


def fit_model(model, X, y, seed):
    if not isinstance(model, MLPRegressor):
        return model.fit(X, y)
    # Keep the original Adam/MSE fit, using in-process batches instead of
    # starting eight DataLoader workers on every epoch and every CV fold.
    torch.manual_seed(seed)
    model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.model.to(model.device).train()
    inputs = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=model.device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=nn_config['lr'])
    for _ in range(nn_config['train_epochs'] + 1):
        order = torch.randperm(len(inputs), device=model.device)
        for batch in order.split(nn_config['batch_size']):
            optimizer.zero_grad()
            prediction = model.model(inputs[batch]).reshape(-1)
            fit_loss = torch.nn.functional.mse_loss(prediction, targets[batch])
            fit_loss.backward()
            optimizer.step()
    model.model.eval()
    return model


def nn_parameter_dict(model, flat_parameters):
    parameters = {}
    offset = 0
    for name, parameter in model.model.named_parameters():
        size = parameter.numel()
        parameters[name] = flat_parameters[offset:offset + size].reshape(parameter.shape)
        offset += size
    return parameters


def nn_curvatures(model, X, y):
    """Return squared-loss gradients, raw Hessian, and mean(g_i g_i.T)."""
    model.model.eval()
    inputs = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=model.device)
    parameters = tuple(model.model.parameters())
    gradients = []
    for i in range(len(inputs)):
        loss_i = (model.model(inputs[i:i + 1]).reshape(()) - targets[i]) ** 2
        gradient = torch.autograd.grad(loss_i, parameters)
        gradients.append(torch.cat([g.reshape(-1) for g in gradient]).detach().cpu().double().numpy())
    gradients = np.stack(gradients)
    empirical_fisher = gradients.T @ gradients / len(inputs)

    # Standard OIC/IJ use the raw Hessian. IFC-OIC uses only empirical_fisher.
    # Differentiate the mean loss in batches to bound temporary Hessian memory.
    flat_parameters = torch.cat([p.detach().reshape(-1) for p in parameters])
    empirical_hessian = np.zeros((len(flat_parameters), len(flat_parameters)))
    for start in range(0, len(inputs), nn_curvature_batch_size):
        batch_X = inputs[start:start + nn_curvature_batch_size]
        batch_y = targets[start:start + nn_curvature_batch_size]

        def mean_loss(flat):
            prediction = functional_call(
                model.model, nn_parameter_dict(model, flat), (batch_X,)).reshape(-1)
            return torch.mean((prediction - batch_y) ** 2)

        batch_hessian = torch.autograd.functional.hessian(
            mean_loss, flat_parameters, vectorize=True)
        empirical_hessian += (batch_hessian.detach().cpu().double().numpy()
                              * len(batch_X) / len(inputs))
    empirical_hessian = (empirical_hessian + empirical_hessian.T) / 2
    return gradients, empirical_hessian, empirical_fisher


def nn_debias(model, X, y, loss, in_sample):
    gradients, empirical_hessian, empirical_fisher = nn_curvatures(model, X, y)
    sample_num = len(X)
    hessian_inv = np.linalg.pinv(empirical_hessian, rcond=nn_pinv_rtol, hermitian=True)
    fisher_inv = np.linalg.pinv(empirical_fisher, rcond=nn_pinv_rtol, hermitian=True)
    active = loss > eta
    # grad h_i = grad fitting_loss_i * 1{loss_i > eta}; no extra factor of two.
    raw_steps = gradients @ hessian_inv
    ifc_steps = gradients @ fisher_inv
    debias = in_sample + np.sum(gradients[active] * raw_steps[active]) / sample_num ** 2
    debias_ifc = in_sample + np.sum(gradients[active] * ifc_steps[active]) / sample_num ** 2

    # Evaluate the IJ parameter perturbation without mutating fitted weights.
    flat_parameters = torch.cat([p.detach().reshape(-1) for p in model.model.parameters()])
    inputs = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    loss_ij = np.zeros(sample_num)
    with torch.no_grad():
        for i in range(sample_num):
            update = torch.as_tensor(raw_steps[i] / sample_num,
                                     dtype=flat_parameters.dtype, device=model.device)
            prediction = functional_call(
                model.model, nn_parameter_dict(model, flat_parameters + update),
                (inputs[i:i + 1],)).item()
            loss_ij[i] = max((y[i] - prediction) ** 2 - eta, 0)
    return debias, debias_ifc, np.mean(loss_ij)


def debias_wine(seed):
    X_tr, y_tr, X_te_list, y_te_list = get_train_test_data(seed)

    if model_class == 'ridge':
        model = Ridge(fit_intercept=False, alpha=alpha0)
        penalty = alpha0
    elif model_class == 'linear':
        model = LinearRegression(fit_intercept=False)
        penalty = 0.0
    elif model_class == 'nn':
        model = make_nn(X_tr.shape[1], seed)
    else:
        raise ValueError("model_class must be 'linear', 'ridge', or 'nn'")
    fit_model(model, X_tr, y_tr, seed)

    y_tr_pred = model.predict(X_tr)
    residual = y_tr - y_tr_pred
    loss = residual ** 2
    sample_num = len(X_tr)
    in_sample = np.mean(np.maximum(loss - eta, 0))
    print('in_sample result', in_sample)

    if model_class == 'nn':
        debias, debias_ifc, debias_ij = nn_debias(model, X_tr, y_tr, loss, in_sample)
        debias0 = debias2 = debias3 = np.nan
        print('OIC result', debias)
        print('IFC-OIC result', debias_ifc)
        print('ALO-IJ result', debias_ij)
        print('First-order LOO, LOO, GCV: N/A for neural networks')
    else:
        # H = (2/n) (X.T @ X + penalty * I), for the squared fitting loss.
        gram_inv = np.linalg.pinv(
            X_tr.T @ X_tr + penalty * np.identity(X_tr.shape[1]), hermitian=True)
        cov_inv = sample_num * gram_inv
        hessian_inv = cov_inv / 2
        # Compute diag(X @ gram_inv @ X.T) without forming an n-by-n matrix.
        hat_diag = np.einsum('ij,ij->i', X_tr @ gram_inv, X_tr)
        if np.any(1 - hat_diag <= np.finfo(float).eps):
            raise ValueError('IFC-OIC and LOO require leverage strictly below one; '
                             'use positive ridge regularization for interpolating fits.')

        grad_interact_sum, grad_interact_sum_ifc = 0.0, 0.0
        loss_ij = np.zeros(sample_num)
        for i in range(sample_num):
            # g_i includes the per-observation penalty alpha/n * ||theta||^2.
            fitting_gradient = (-2 * residual[i] * X_tr[i]
                                + 2 * penalty / sample_num * model.coef_)
            influence_step = hessian_inv @ fitting_gradient / sample_num
            if loss[i] > eta:
                grad_interact = -2 * residual[i] * X_tr[i]
                interaction = grad_interact.dot(influence_step) / sample_num
                grad_interact_sum -= interaction
                # IFC-OIC replaces H by (1 - h_i) H, and nothing else.
                grad_interact_sum_ifc -= interaction / (1 - hat_diag[i])
            # CV/LOO retain sklearn's fixed alpha. Delete only the observation's
            # squared loss, retaining the ridge penalty, for the matching IJ step.
            ij_update = -gram_inv @ X_tr[i] * residual[i]
            residual_ij = residual[i] - X_tr[i].dot(ij_update)
            loss_ij[i] = max(residual_ij ** 2 - eta, 0)

        debias = in_sample - grad_interact_sum
        debias_ifc = in_sample - grad_interact_sum_ifc
        debias_ij = np.mean(loss_ij)
        print('OIC result', debias)
        print('IFC-OIC result', debias_ifc)
        print('ALO-IJ result', debias_ij)

        # Retain the original analytical comparisons, with explicit output labels.
        loss_alo_1 = loss + 2 * residual ** 2 * hat_diag / (1 - hat_diag)
        debias0 = np.mean(np.maximum(loss_alo_1 - eta, 0))
        print('First-order LOO result', debias0)

        # Exact fixed-alpha linear/ridge LOO, conditional on this feature matrix.
        loss_adj = loss / (1 - hat_diag) ** 2
        debias2 = np.mean(np.maximum(loss_adj - eta, 0))
        print('LOO result', debias2)

        debias3 = np.mean(np.maximum(
            loss / (1 - np.mean(hat_diag)) ** 2 - eta, 0))
        print('GCV result', debias3)

    cv_result = np.nan
    if CV_compare:
        # As in the source, evaluate CV conditional on the transformed design.
        kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
        cv_losses = np.zeros(sample_num)
        for fold, (train_label, test_label) in enumerate(kf.split(X_tr)):
            if model_class == 'ridge':
                model1 = Ridge(fit_intercept=False, alpha=alpha0)
            elif model_class == 'nn':
                model1 = make_nn(X_tr.shape[1], seed + fold + 1)
            else:
                model1 = LinearRegression(fit_intercept=False)
            fit_model(model1, X_tr[train_label], y_tr[train_label], seed + fold + 1)
            cv_residual = y_tr[test_label] - model1.predict(X_tr[test_label])
            cv_losses[test_label] = np.maximum(cv_residual ** 2 - eta, 0)
        cv_result = np.mean(cv_losses)
        print('CV result', cv_result)

    # OOS is evaluated before any parameter mutation (IJ above is non-mutating).
    out_sample = np.zeros(len(X_te_list))
    for i in range(len(X_te_list)):
        y_te_pred = model.predict(X_te_list[i])
        loss_te = (y_te_pred - y_te_list[i]) ** 2
        out_sample[i] = np.mean(np.maximum(loss_te - eta, 0))
    out_sample_mean = np.average(out_sample, weights=[len(y) for y in y_te_list])
    print('OOS result', out_sample_mean)
    print('==========')
    return (in_sample, debias, debias_ifc, cv_result, debias_ij,
            debias0, debias2, debias3, out_sample_mean)


if __name__ == '__main__':
    result = np.zeros((loop_num, len(RESULT_NAMES)))
    for i in range(loop_num):
        result[i] = debias_wine(i * 5)
    print('model class:', model_class, 'alpha', alpha0, 'poly size', poly_size)
    print('train frac', train_frac)
    print(', '.join(RESULT_NAMES))
    print('avg', np.mean(result, axis=0),
          'standard error', np.std(result, axis=0, ddof=1) / np.sqrt(loop_num))
