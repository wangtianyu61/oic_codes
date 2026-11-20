# mean-variance portfolio allocation
import numpy as np
import math
import cvxpy as cp

from sklearn.model_selection import KFold
from sklearn.utils import resample
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
import torch
from scipy.stats import chi2, multivariate_normal
from sklearn.neighbors import KernelDensity



class mv_portfolio:
    def __init__(self, return_mean, return_cov, lambda_1, lambda_2):
        self.return_mean = return_mean
        self.return_cov = return_cov
        self.port_num = self.return_cov.shape[0]
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        # the C in the \eps = C / n
        self.ambiguity_coef = 3
        self.eps = 0.001
        self.true_sol()
    
    def sol_obj(self, sol):
        return np.dot(np.dot(self.return_cov, sol), sol) + self.lambda_2 * np.dot(sol, sol) - self.lambda_1 * np.dot(self.return_mean, sol)
    
    def generate(self, sample_num):
        sample_data = np.random.multivariate_normal(self.return_mean, self.return_cov, size = sample_num)
        return sample_data
    def true_sol(self):
        result_array = 2 * (self.return_cov + self.lambda_2 * np.identity(self.return_cov.shape[0]))
        self.true_sol_val = np.dot(np.linalg.inv(result_array), self.lambda_1 * self.return_mean)

    def normal_idpd_pf(self, sample_data):
        mu = np.mean(sample_data, axis = 0)
        sigma2 = np.diagonal(np.cov(sample_data.T))
        return mu, sigma2



    def normal_idpd_opt(self, sample_data):
        """
        solution of ETO under normal distributon
        """
        mu, sigma2 = self.normal_idpd_pf(sample_data)

        sample_num, __ = sample_data.shape
        
        sample_data = np.random.multivariate_normal(mu, np.diag(sigma2), 100 * sample_num)
        pf_sol = self.saa_sol(sample_data)

        return pf_sol

    def normal_pf_eval(self, sol, sample_data):
        mu, sigma2 = self.normal_idpd_pf(sample_data)

        sample_num, __ = sample_data.shape
        mc_data = np.random.multivariate_normal(mu, np.diag(sigma2), 100 * sample_num)
        return self.sample_eval(sol, mc_data)

    def normal_idpd_pf_correction(self, sol, sample_data, type = 'empirical'):
        """
        bias of ETO under normal distribution
        - empirical: base estimator is the empirical estimator
        - parametric: base estimator is the estimator under the parametric distribution
        """
        mu, sigma2 = self.normal_idpd_pf(sample_data)
        sample_num, __ = sample_data.shape
        # compute the covariance matrix of the asymptotic parametric distribution
        ## asymptotic variance of empirical variance estimator
        Var_p = list(sigma2) + list([np.mean([(sample_data[j][i] - mu[i])** 4 for j in range(sample_num)]) - sigma2[i]**2 for i in range(self.port_num)])

        Phi_p = torch.diag(torch.from_numpy(np.array(Var_p)))
        
        x_input = torch.from_numpy(sol)
        mu_input = torch.from_numpy(mu)
        sigma2_input = torch.from_numpy(sigma2)
        dimx = self.port_num

        jacob_x_p = []
        hes_x_p = []
        for index in range(dimx):
            def func_1(params):
                mu = params[0:dimx]
                sigma2 = params[dimx:]
                delta_mu = mu - torch.from_numpy(self.return_mean)
                cov_decomp = torch.diag(sigma2) + torch.mm(delta_mu.reshape(-1, 1), delta_mu.reshape(1, -1)) + torch.from_numpy(self.lambda_2 * np.identity(self.port_num))
                return self.lambda_1 * torch.linalg.solve(cov_decomp, mu)[index] / 2
            
            params = torch.cat((mu_input, sigma2_input))
            hes = hessian(func_1, params)
            jacob = jacobian(func_1, params)

            jacob_x_p.append(jacob)
            hes_x_p.append(hes)

        ## compute hes_h_p
        if type == 'parametric':
            loop_num = 5 * sample_num
        else:
            loop_num = sample_num
        gradient_interact = 0
        sample_hes_h_p = 0

        for i in range(loop_num):
            if type == 'parametric':
                rv = torch.normal(mean = mu_input, std = torch.sqrt(sigma2_input))
            elif type == 'empirical':
                rv = torch.from_numpy(sample_data[i % sample_num])
            else:
                raise NotImplementedError
            delta_rv = rv - torch.from_numpy(self.return_mean)
            COV = torch.mm(delta_rv.reshape(-1, 1), delta_rv.reshape(1, -1))
            def cost_func(x):
                return  torch.inner(x, torch.inner(COV, x)) - self.lambda_1 * torch.inner(rv, x)  + self.lambda_2 * torch.inner(x, x)
            hes_h_x = hessian(cost_func, x_input)
            jacob_h_x = jacobian(cost_func, x_input)

            IF = list([rv[j] - mu[j] for j in range(self.port_num)]) + list([(rv[j] - mu[j])**2 - sigma2[j] for j in range(self.port_num)])
            IF = torch.from_numpy(np.array(IF))
            sample_jacob_h_p = 0
            for j in range(dimx):
                sample_jacob_h_p += jacob_h_x[j] * jacob_x_p[j]
        # used for empirical debiasing
            gradient_interact += torch.inner(sample_jacob_h_p, IF) / (loop_num ** 2)


            for j in range(dimx):
                sample_hes_h_p += (hes_h_x[j][j] * torch.mm(jacob_x_p[j].reshape(-1, 1), jacob_x_p[j].reshape(1, -1)) + jacob_h_x[j] * hes_x_p[j])
                for k in range(j, dimx):
                    sample_hes_h_p += 2 * hes_h_x[j][k] * torch.mm(jacob_x_p[j].reshape(-1, 1), jacob_x_p[k].reshape(1, -1))

        I_h_p = sample_hes_h_p/loop_num
        # here the esimators we apply are all unbiased, so there is no higher-order bias...

        
        if type == 'parametric':
            return torch.trace(torch.mm(I_h_p, Phi_p)).numpy() / (2 * sample_num)
        elif type == 'empirical':
            return -gradient_interact.numpy()
            
    def chi2_DRO_sol(self, sample_data):
        """
        solution of chi2 dro
        """

        sample_size, __ = sample_data.shape
        weight = cp.Variable(self.port_num)
        eta = cp.Variable()
        aux_s = cp.Variable(sample_size)

        loss = math.sqrt((1 + self.ambiguity_coef / sample_size) /sample_size) * cp.norm(cp.pos(aux_s - eta), 2) + eta

        cons = []
        for i in range(sample_size):
            product_term = sample_data[i] - self.return_mean
            cons += [cp.quad_form(weight, np.outer(product_term, product_term) + self.eps * np.identity(self.port_num)) + self.lambda_2 * cp.norm(weight, 2) ** 2 <= aux_s[i] + self.lambda_1 * weight @ sample_data[i]]
        
        prob = cp.Problem(cp.Minimize(loss), cons)
        prob.solve(solver = cp.MOSEK)

        return weight.value

    def chi2dro_sol_correction(self, sol, sample_data):
        """
        bias of chi2 dro
        """
        return self.saa_correction(sol, sample_data)
    
    def saa_sol(self, sample_data):
        """
        solution of saa
        """
        sample_mean = np.mean(sample_data, axis = 0)
        cov_info = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[1]))
        for i in range(sample_data.shape[0]):
            product_term = sample_data[i] - self.return_mean
            cov_info[i] = np.dot(product_term.reshape(-1, 1), product_term.reshape(1, -1))
        sample_cov = np.mean(cov_info, axis = 0) 


        return_array = 2 * (sample_cov + self.lambda_2 * np.identity(sample_cov.shape[0]))
        
        self.saa_sol_val = np.dot(np.linalg.inv(return_array), self.lambda_1 * sample_mean)
        return self.saa_sol_val
    


    def uniform_sol(self, sample_data):
        """
        solution of SAA-A
        """

        sample_mean = np.mean(sample_data, axis = 0)
        cov_info = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[1]))
        for i in range(sample_data.shape[0]):
            product_term = sample_data[i] - self.return_mean
            cov_info[i] = np.dot(product_term.reshape(-1, 1), product_term.reshape(1, -1))
        sample_cov = np.mean(cov_info, axis = 0)

        


        theta = self.lambda_1 * np.sum(sample_mean) / (2 * (np.sum(sample_cov) + self.lambda_2 * self.port_num))
        self.uniform_sol_val = theta * np.ones(self.port_num)
        return self.uniform_sol_val

    
    def block_sol(self, sample_data):
        """
        solution of SAA-B
        """

        sample_mean = np.mean(sample_data, axis = 0)
        cov_info = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[1]))
        sample_size, __ = sample_data.shape
        weight = cp.Variable(self.port_num)

        for i in range(sample_data.shape[0]):
            product_term = sample_data[i] - self.return_mean
            cov_info[i] = np.dot(product_term.reshape(-1, 1), product_term.reshape(1, -1))
        sample_cov = np.mean(cov_info, axis = 0)

        loss = cp.quad_form(weight, sample_cov) - self.lambda_1 * sample_mean @ weight + self.lambda_2 * cp.norm(weight, 2) ** 2
        cons = []
        for i in range(1, int(self.port_num / 2)):
            cons += [weight[0] == weight[i]]
        for i in range(int(self.port_num / 2), self.port_num):
            cons += [weight[int(self.port_num / 2)] == weight[i]]
        prob = cp.Problem(cp.Minimize(loss), cons)
        prob.solve(solver = cp.MOSEK)

        return weight.value

    def uniform_sol_KDE(self, sample_data):
        sample_num, port_num = sample_data.shape
        h_n = 2 * sample_num ** (-1/(port_num + 4))
        kde = KernelDensity(kernel = 'gaussian', bandwidth = h_n).fit(sample_data)
        mc_num = 10000
        mc_data = kde.sample(mc_num)
        return self.uniform_sol(mc_data)


    def uniform_correction_KDE(self, sol, sample_data):
        sample_num, port_num = sample_data.shape
        assert (port_num <= 3)

        h_n = 2 * sample_num ** (-1/(port_num + 4))
        kde = KernelDensity(kernel = 'gaussian', bandwidth = h_n).fit(sample_data)
        import itertools
        lw, up = np.min(sample_data, axis = 0), np.max(sample_data, axis = 0)
        #create sample grid for finite-integral simulation
        mc_int_dim_num = 100
        mc_int_num = mc_int_dim_num ** port_num
        

        if port_num == 1:
            rv = np.arange(start = lw[0], stop = up[0], step = (up[0] - lw[0]) / mc_int_dim_num)
        elif port_num == 2:
            rv0 = np.arange(start = lw[0], stop = up[0], step = (up[0] - lw[0]) / mc_int_dim_num)
            rv1 = np.arange(start = lw[1], stop = up[1], step = (up[1] - lw[1]) / mc_int_dim_num)
            rv = list([np.array(item) for item in itertools.product(rv0, rv1)])
        else:
            rv0 = np.arange(start = lw[0], stop = up[0], step = (up[0] - lw[0]) / mc_int_dim_num)
            rv1 = np.arange(start = lw[1], stop = up[1], step = (up[1] - lw[1]) / mc_int_dim_num)
            rv2 = np.arange(start = lw[2], stop = up[2], step = (up[2] - lw[2]) / mc_int_dim_num)
            rv = list([np.array(item) for item in itertools.product(rv0, rv1, rv2)])

        gradient_interact = 0
        for i in range(sample_num):
            int_val = 0
            for k in range(mc_int_num):
                var = multivariate_normal(mean = sample_data[i], cov = h_n * np.identity(port_num))
                rv_matrix = np.dot((rv[k] - self.return_mean).reshape(-1, 1), (rv[k] - self.return_mean).reshape(1, -1))
                int_val -= (var.pdf(rv[k]) - math.exp(kde.score_samples(rv[k].reshape(1, -1)))) * \
                            ((np.sum(rv_matrix) + self.lambda_2 * port_num) * 2 * sol[0] - self.lambda_1 * np.sum(rv[k])) / mc_int_num

            rv_matrix = np.dot((sample_data[i] - self.return_mean).reshape(-1, 1), (sample_data[i] - self.return_mean).reshape(1, -1))
            gradient = (np.sum(rv_matrix) + self.lambda_2 * port_num) * 2 * sol[0] - self.lambda_1 * np.sum(sample_data[i])
            gradient_interact += gradient * int_val

        hessian_val = 0
        for k in range(mc_int_num):
            rv_matrix = np.dot((rv[k] - self.return_mean).reshape(-1, 1), (rv[k] - self.return_mean).reshape(1, -1))
            hessian_val += 2 * (np.sum(rv_matrix) + self.lambda_2 * port_num)/ mc_int_num

        return gradient_interact/ (hessian_val * sample_num ** 2)





    def sample_eval(self, sol, sample_data):
        sample_mean = np.mean(sample_data, axis = 0)
        cov_info = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[1]))
        for i in range(sample_data.shape[0]):
            product_term = sample_data[i] - self.return_mean
            cov_info[i] = np.dot(product_term.reshape(-1, 1), product_term.reshape(1, -1))
        sample_cov = np.mean(cov_info, axis = 0)

        return np.dot(sample_cov, sol).dot(sol) - self.lambda_1 * np.dot(sample_mean, sol) + self.lambda_2 * np.dot(sol, sol)



    def true_eval(self, sol):
        return np.dot(self.return_cov, sol).dot(sol) - self.lambda_1 * np.dot(self.return_mean, sol) + self.lambda_2 * np.dot(sol, sol)



    def uniform_correction(self, sol, sample_data):
        """
        bias correction of SAA-A
        """

        sample_mean = np.mean(sample_data, axis = 0)
        diff = sample_data - self.return_mean
        diff_compute = np.zeros(sample_data.shape[0])
        for i in range(sample_data.shape[0]):
            diff_compute[i] = np.sum(np.dot(diff[i].reshape(-1, 1), diff[i].reshape(1, -1))) + self.lambda_2 * self.port_num
        I_h_inv = 1/np.mean(2 * diff_compute)
        J_h = np.zeros(sample_data.shape[0])
        for i in range(sample_data.shape[0]):
            J_h[i] = (2 * sol[0] * diff_compute[i] - self.lambda_1 * np.sum(sample_data[i])) ** 2
        return I_h_inv * np.mean(J_h) / (sample_data.shape[0])
        #return I_h_inv, np.mean(J_h)


    # we use torch to store the data
    def IO_correction(self, I_h, J_h, sample_num):
        bias_correction = torch.trace(torch.linalg.solve(I_h, J_h))
        return bias_correction.numpy() / sample_num



    def uniform_correction_torch(self, sol, sample_data):
        sample_num, __ = sample_data.shape
        I_h, J_h  = 0, 0
        x_input = torch.from_numpy(sol[0:1])
        for i in range(sample_num):
            def cost_func(x):
                diff = sample_data[i] - self.return_mean

                return (np.sum(np.dot(diff.reshape(-1, 1), diff.reshape(1, -1))) + self.lambda_2 * self.port_num) * torch.square(x) \
                        - self.lambda_1 * np.sum(sample_data[i]) * x 
            hes_h_x = hessian(cost_func, x_input)
            jacob_h_x = jacobian(cost_func, x_input)

            I_h += hes_h_x 
            J_h += torch.mm(jacob_h_x.reshape(-1, 1), jacob_h_x.reshape(1, -1))
        return self.IO_correction(I_h / sample_num, J_h / sample_num, sample_num)
    
    def block_correction_torch(self, sol, sample_data):
        sample_num, __ = sample_data.shape
        I_h, J_h = 0, 0
        x_input = torch.from_numpy(np.array((sol[0], sol[int(self.port_num / 2)])))
        for i in range(sample_num):
            def cost_func(x):
                new_x = torch.cat((x[0] * torch.ones(int(self.port_num / 2)), x[1] * torch.ones(int(self.port_num / 2))))
                rv = torch.from_numpy(sample_data[i]).float()
                diff = rv - torch.from_numpy(self.return_mean)
                cov_diff = torch.mm(diff.reshape(-1, 1), diff.reshape(1, -1)).float()
                #).double()
                #print((np.dot(diff.reshape(-1, 1), diff.reshape(1, -1))).dtype)
                #print(cov_diff.type(), rv.type())
                return torch.inner(torch.inner(cov_diff, new_x), new_x) + self.lambda_2 * torch.sum(torch.square(new_x)) - self.lambda_1 * torch.dot(rv, new_x)


            hes_h_x = hessian(cost_func, x_input)
            jacob_h_x = jacobian(cost_func, x_input)
        
            I_h += hes_h_x
            J_h += torch.mm(jacob_h_x.reshape(-1, 1), jacob_h_x.reshape(1, -1))
        
        return self.IO_correction(I_h / sample_num, J_h / sample_num, sample_num)


    def uniform_cover_correction_compute(self, sol, sample_data):
        I_h_inv, J_h = self.uniform_correction(sol, sample_data)
        lw_bd, up_bd = chi2.ppf(0.005, 1), chi2.ppf(0.995, 1)
        base = self.sample_eval(sol, sample_data)
        base_correct = base + I_h_inv * J_h / (2 * sample_data.shape[0])
        cover_0, cover_1 = 0, 0
        # without correction
        if self.true_eval(sol) >= base + lw_bd * I_h_inv * J_h / sample_data.shape[0] \
            and self.true_eval(sol) <= base + up_bd * I_h_inv * J_h / sample_data.shape[0]:
                cover_0 = 1
        # with correction
        #+ lw_bd * I_h_inv * J_h / sample_data.shape[0] 
        if self.true_eval(sol) >= base_correct \
            and self.true_eval(sol) <= base_correct + 5 * up_bd * I_h_inv * J_h / sample_data.shape[0]:
                cover_1 = 1
        return cover_0, cover_1

    def saa_correction(self, sol, sample_data):
        sample_mean = np.mean(sample_data, axis = 0)
        diff = sample_data - self.return_mean
        diff_compute = np.zeros((sample_data.shape[0], self.port_num, self.port_num))
        for i in range(sample_data.shape[0]):
            diff_compute[i] = np.dot(diff[i].reshape(-1, 1), diff[i].reshape(1, -1)) + self.lambda_2 * np.identity(self.port_num)
        I_h_inv = np.linalg.inv(np.mean(diff_compute, axis = 0)) / 2
        J_h = np.zeros((sample_data.shape[0], self.port_num, self.port_num))
        for i in range(sample_data.shape[0]):
            product_term = 2 * np.dot(diff_compute[i], sol) - self.lambda_1 * sample_data[i]
            J_h[i] = np.dot(product_term.reshape(-1, 1), product_term.reshape(1, -1))
        J_h_final = np.mean(J_h, axis = 0)
        bias = np.dot(I_h_inv, J_h_final)
        #return I_h_inv, J_h_final
        return bias.trace()/ (sample_data.shape[0])

    def saa_cover_correction_compute(self, sol, sample_data):
        simulate_size = 10000
        I_h_inv, J_h = self.saa_correction(sol, sample_data)
        I_h = np.linalg.inv(I_h_inv)

        # simulation to find the upper and lower 
        cov = np.dot(I_h_inv, J_h).dot(I_h_inv) / sample_data.shape[0]
        seq = np.random.multivariate_normal(np.zeros(sample_data.shape[1]), cov, simulate_size)
        simulate_output = np.zeros(simulate_size)
        for i in range(simulate_size):
            simulate_output[i] = np.dot(seq[i], I_h).dot(seq[i])
        ## find quantile
        lw_bd = np.quantile(simulate_output, 0.005)
        up_bd = np.quantile(simulate_output, 0.995)

        base = self.sample_eval(sol, sample_data)
        base_correct = base + np.dot(I_h_inv, J_h).trace()/(2 * sample_data.shape[0])
        cover_0, cover_1 = 0, 0
        # without correction
        if self.true_eval(sol) >= base + lw_bd \
            and self.true_eval(sol) <= base + up_bd:
                cover_0 = 1
        # with correction
        if self.true_eval(sol) >= base_correct + lw_bd \
            and self.true_eval(sol) <= base_correct + up_bd:
                cover_1 = 1
        return cover_0, cover_1
    
    def jackknife_debias(self, sample_data, method):
        sample_num = sample_data.shape[0]
        kf = KFold(n_splits = sample_num)
        eval_perform = np.zeros(sample_num)
        i = 0
        for (train_index, __) in kf.split(sample_data):
            train_data = sample_data[train_index]
            if method == 'saa':
                sol = self.saa_sol(train_data)
            elif method == 'normal':
                sol = self.normal_idpd_opt(train_data)
            elif method == 'uniform':
                sol = self.uniform_sol(train_data)
            elif method == 'block':
                sol = self.block_sol(train_data)
            elif method == 'chi2dro':
                sol = self.chi2_DRO_sol(train_data)
            else:
                raise NotImplementedError
            eval_perform[i] = self.sample_eval(sol, train_data)
            i += 1
        return np.mean(eval_perform)
    
    def bootstrap_debias(self, sample_data, btsp_num = 10, method = 'SAA'):
        sample_num = len(sample_data)
        eval_array = np.zeros(btsp_num)
        for i in range(btsp_num):
            resample_data = resample(sample_data, replace = True, n_samples = sample_num)
            if method == 'saa':
                sol = self.saa_sol(resample_data)
            elif method == 'normal':
                sol = self.normal_idpd_opt(resample_data)
            elif method == 'uniform':
                sol = self.uniform_sol(resample_data)
            elif method == 'block':
                sol = self.block_sol(resample_data)
            elif method == 'chi2dro':
                sol = self.chi2_DRO_sol(resample_data)
            else:
                raise NotImplementedError
            eval_array[i] = self.sample_eval(sol, resample_data)
        return np.mean(eval_array)
    
    def CV_performance(self, sample_data, fold_num = 3, method = 'SAA'):
        kf = KFold(n_splits = fold_num)
        model_val = 0
        for (train_index, test_index) in kf.split(sample_data):
            train_data = sample_data[train_index]
            test_data = sample_data[test_index]
            #choose the sol
            if method == 'saa':
                cv_sol = self.saa_sol(train_data)
            elif method == 'normal':
                cv_sol = self.normal_idpd_opt(train_data)
            elif method == 'uniform':
                cv_sol = self.uniform_sol(train_data)
            elif method == 'block':
                cv_sol = self.block_sol(train_data)
            elif method == 'chi2dro':
                cv_sol = self.chi2_DRO_sol(train_data)
            else:
                raise NotImplementedError         
            ## OOS test
            model_val += self.sample_eval(cv_sol, test_data) / fold_num
        return model_val
