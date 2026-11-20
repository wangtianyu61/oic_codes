import numpy as np
import math
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from scipy.stats import norm
from sklearn.utils import resample


# univariate newsvendor problem
class newsvendor:
    def __init__(self, price, cost, dist_type = 'exponential'):
        assert (price > cost)
        self.p = price
        self.c = cost
        self.dist_type = dist_type
        self.approx = False
    def generate(self, sample_num):
        if self.dist_type == 'normal':
            mu = 100
            sigma = 40
            temp_data = np.random.normal(mu, sigma, sample_num) + np.random.uniform(-5, 5, sample_num)
            #temp_data[temp_data < 0] = 0
            return temp_data
        elif self.dist_type == 'exponential':
            mu = 100
            temp_data = np.random.exponential(mu, sample_num) + np.random.uniform(-5, 5, sample_num)
            return temp_data
        elif self.dist_type == 'mixed':
            self.dist_type = 'normal'
            A1 = self.generate(sample_num) * (1/2)
            self.dist_type = 'exponential'
            A2 = self.generate(sample_num) * (1/2)
            #recover
            self.dist_type = 'mixed'
            return A1 + A2 + np.random.uniform(-5, 5, sample_num)
        else:
            raise NotImplementedError


    def saa(self, sample_data):
        """
        solution of saa
        """
        return np.percentile(sample_data, 100 * (1 - self.c / self.p))
    
    def saa_correction(self, sol, sample_data):
        """
        bias of saa
        """
        sample_num = sample_data.shape[0]
        sample_data = sample_data.reshape(-1, 1)
        #rule-of-thumb bandwidth estimator
        if self.approx == False:
            h_n = 1.06 * np.std(sample_data) * sample_num ** (-0.2)
            kde = KernelDensity(kernel = 'gaussian', bandwidth = h_n).fit(sample_data)
            log_dens = kde.score_samples(sol.reshape(1, -1))
            #print(sol, log_dens)
            return self.c * (self.p - self.c) / (1 * self.p * math.exp(log_dens) * sample_num) 
        #* np.exp(log_dens))
        else:
            approx_hessian = 0
            for i in range(sample_num):
                approx_hessian += self.approx_hessian(sol, sample_data[i]) / sample_num
        
            h_n = 1.06 * np.std(sample_data) * sample_num ** (-0.2)
            kde = KernelDensity(kernel = 'gaussian', bandwidth = h_n).fit(sample_data)
            log_dens = kde.score_samples(sol.reshape(1, -1))
            
            return approx_hessian * (self.p - self.c) * self.c / (sample_num * math.exp(log_dens) ** 2 * self.p ** 2)

    def normal(self, sample_data):
        """
        solution of ETO under normal distribution
        """
        mu, sigma = np.mean(sample_data), np.std(sample_data)
        return mu + sigma * norm.ppf(1 - self.c / self.p)


    def normal_correction(self, sol, sample_data):
        """
        bias of ETO under normal distribution
        """

        mu, sigma = np.mean(sample_data), np.std(sample_data)
        sample_num = sample_data.shape[0]

        gradient_interact = 0
        for i in range(sample_num):
            if sample_data[i] < sol:
                gradient = self.c
            else:
                gradient = self.c - self.p
            IF = np.array([sample_data[i] - mu, (sample_data[i] - mu) ** 2 - sigma ** 2])
            partial_x_p = np.array([1, norm.ppf(1 - self.c / self.p)/(2 * sigma)])
            gradient_interact += gradient * np.dot(partial_x_p, IF)
        return -gradient_interact/(sample_num ** 2)

    def normal_correction_MS(self, sol, sample_data):
        """
        bias estimator of "Data-driven profit estimation error in the newsvendor model", 
        which assumes the underlying distribution is normal.
        """
        mu, sigma = np.mean(sample_data), np.std(sample_data)
        quantile_val = norm.ppf(1 - self.c / self.p)
        return -self.p * sigma * (2 + quantile_val ** 2) * norm.pdf(quantile_val) / (4 * len(sample_data))
    

    def normal_eval(self, normal_sol, sample_data):
        mu, sigma = np.mean(sample_data), np.std(sample_data)
        sample_num = len(sample_data) * 20
        temp_data = np.random.normal(mu, sigma, sample_num)
        return self.empirical_eval(normal_sol, temp_data)

    def exp_eval(self, exp_sol, sample_data):
        mu = np.mean(sample_data)
        temp_data = np.random.exponential(mu, len(sample_data) * 20)
        return self.empirical_eval(exp_sol, temp_data)


    def exp(self, sample_data):
        """
        solution of ETO under normal distribution
        """
        mu = np.mean(sample_data)
        return math.log(self.p / self.c) * mu
        
    def approx_gradient(self, sol, rv):
        #use the smooth function to approximate
        limit_m = 200
        try:
            gradient = (self.c - self.p) + self.p / (1 + math.exp(-limit_m * (sol - rv)))
            return gradient
        except Exception as e:
            if sol > rv:
                return self.c
            else:
                return self.c - self.p

    def approx_hessian(self, sol, rv):
        limit_m = 1
        try:
            hessian = 1 * self.p /((math.exp(-limit_m * (sol - rv)/2) + math.exp(limit_m * (sol - rv)/2))/math.sqrt(limit_m))**2
            return hessian
        except Exception as e:
            return 0
    


    def exp_correction(self, sol, sample_data):
        """
        bias of ETO under normal distribution
        """
        mu = np.mean(sample_data)
        sample_num = sample_data.shape[0]
        gradient_interact = 0

        for i in range(sample_num):
            if self.approx == False:
                if sample_data[i] < sol:
                    gradient = self.c
                else:
                    gradient = self.c - self.p
            else:
                gradient = self.approx_gradient(sol, sample_data[i])

            gradient_interact += gradient * math.log(self.p / self.c) * (sample_data[i] - mu)
        return -gradient_interact/(sample_num ** 2)
    
    def exp_correction_MS(self, sol, sample_data):
        """
        bias estimator of "Data-driven profit estimation error in the newsvendor model", 
        which assumes the underlying distribution is normal.
        """
        return  -self.c * sol * (math.log(self.p / self.c))** 2/(2 * len(sample_data))

    
    def OS_exp(self, sample_data):
        """
        solution of operational statistics under exponential distribution
        """
        mu = np.mean(sample_data)
        sample_num = len(sample_data)
        return sample_num * ((self.p / self.c) ** (1/(sample_num + 1)) - 1) * mu

    def OS_exp_correction(self, sol, sample_data):
        """
        bias of operational statistics under exponential distribution
        """
        mu = np.mean(sample_data)
        sample_num = sample_data.shape[0]
        gradient_interact = 0

        for i in range(sample_num):
            if self.approx == False:
                if sample_data[i] < sol:
                    gradient = self.c
                else:
                    gradient = self.c - self.p
            else:
                gradient = self.approx_gradient(sol, sample_data[i])

            gradient_interact += gradient * sample_num * ((self.p / self.c) ** (1/(sample_num + 1)) - 1) * (sample_data[i] - mu)
        return -gradient_interact/(sample_num ** 2)
    
    def KDE(self, sample_data):
        sample_num = sample_data.shape[0]
        sample_data = sample_data.reshape(-1, 1)
        #rule-of-thumb bandwidth estimator
        h_n = 1.06 * np.std(sample_data) * sample_num ** (-0.2)
        kde = KernelDensity(kernel = 'gaussian', bandwidth = h_n).fit(sample_data)
        mc_num = 10000
        mc_data = kde.sample(mc_num)
        return np.percentile(mc_data, 100 * (1 - self.c / self.p))


    def KDE_correction(self, sol, sample_data):
        lw, up = np.min(sample_data), np.max(sample_data)
        sample_num = sample_data.shape[0]
        sample_data = sample_data.reshape(-1, 1)
        #rule-of-thumb bandwidth estimator
        h_n = 1.06 * np.std(sample_data) * sample_num ** (-0.2)
        kde = KernelDensity(kernel = 'gaussian', bandwidth = h_n).fit(sample_data)
        mc_int_num = 1000
        gradient_interact = 0
        rv = np.arange(start = lw, stop = up, step = (up - lw) / mc_int_num)
        for i in range(sample_num):
            int_val = 0
            for k in range(mc_int_num):
                inner_val = norm.pdf(rv[k], sample_data[i], h_n) - math.exp(kde.score_samples(rv[k].reshape(1, -1)))
                if sol < rv[k]:
                    int_val -= self.c * inner_val/ mc_int_num
                else:
                    int_val += (self.p - self.c) * inner_val/ mc_int_num

            if sol < sample_data[i]:
                gradient_interact += (self.c - self.p) * int_val
            else:
                gradient_interact += self.c * int_val



        return -gradient_interact/(sample_num ** 2 * self.p * math.exp(kde.score_samples(sol.reshape(1, -1))))

    def true_eval(self, sol):
        """
        true performance of one solution
        """
        mc_num = 50000
        mc_data = self.generate(mc_num)
        return self.empirical_eval(sol, mc_data) 


    def empirical_eval(self, sol, sample_data):
        sample_num = sample_data.shape[0]
        cost = self.c * sol
        for i in range(sample_num):
            cost -= self.p * min(sample_data[i], sol) / sample_num
        return cost

    def CV_performance(self, sample_data, fold_num = 3, method = 'SAA'):
        kf = KFold(n_splits = fold_num)
        model_val = 0
        for (train_index, test_index) in kf.split(sample_data):
            train_data = sample_data[train_index]
            test_data = sample_data[test_index]
            #choose the sol
            if method == 'saa':
                cv_sol = self.saa(train_data)
            elif method == 'normal':
                cv_sol = self.normal(train_data)
            elif method == 'exponential':
                cv_sol = self.exp(train_data)
            elif method == 'exp_OS':
                cv_sol = self.OS_exp(train_data)
            else:
                raise NotImplementedError            
            ## OOS test
            model_val += self.empirical_eval(cv_sol, test_data) / fold_num
        return model_val
    
    def bootstrap_debias(self, sample_data, btsp_num = 10, method = 'SAA'):
        sample_num = len(sample_data)
        eval_array = np.zeros(btsp_num)
        for i in range(btsp_num):
            resample_data = resample(sample_data, replace = True, n_samples = sample_num)
            if method == 'saa':
                sol = self.saa(resample_data)
            elif method == 'normal':
                sol = self.normal(resample_data)
            elif method == 'exponential':
                sol = self.exp(resample_data)
            elif method == 'exp_OS':
                sol = self.OS_exp(resample_data)
            else:
                raise NotImplementedError
            eval_array[i] = self.empirical_eval(sol, resample_data)
        return np.mean(eval_array), np.std(eval_array)

    def bootstrap_difference_debias(self, sample_data, btsp_num = 10, method = ['SAA', 'normal']):
        sample_num = len(sample_data)
        eval_array = np.zeros(btsp_num)
        for i in range(btsp_num):
            resample_data = resample(sample_data, replace = True, n_samples = sample_num)
            if method == 'SAA':
                sol = self.saa(resample_data)
            eval_array[i] = self.empirical_eval(sol, resample_data)
        return np.mean(eval_array), np.std(eval_array)


    def direct_std(self, sample_data, sol1, sol2 = None, base = None):
        """
        If sol2 is None, it means that we are calculating the s.d. of one given solution;
        Otherwise, it means that we are calculating the s.d. of the difference of two data-driven sols.
        """
        sample_num = sample_data.shape[0]
        sample_var = np.zeros(sample_num)
        for i in range(sample_num):
            sample_var[i] = self.empirical_eval(sol1, sample_data[i:(i + 1)]) - self.empirical_eval(sol2, sample_data[i:(i + 1)])

        if base == None:
            return np.std(sample_var) / math.sqrt(sample_num)
        else:
            return math.sqrt(np.mean((sample_var - base)**2)) / math.sqrt(sample_num)


    def jackknife_debias(self, sample_data, method = 'SAA'):
        sample_num = sample_data.shape[0]
        kf = KFold(n_splits = sample_num)
        eval_perform = np.zeros(sample_num)
        i = 0
        for (train_index, __) in kf.split(sample_data):
            train_data = sample_data[train_index]
            if method == 'saa':
                sol = self.saa(train_data)
            elif method == 'normal':
                sol = self.normal(train_data)
            elif method == 'exponential':
                sol = self.exp(train_data)
            elif method == 'exp_OS':
                sol = self.OS_exp(train_data)
            else:
                raise NotImplementedError
            eval_perform[i] = self.empirical_eval(sol, train_data)
            i += 1
        return np.mean(eval_perform), np.std(eval_perform) * math.sqrt(sample_num - 1)
    
    def jackknife_difference_debias(self, sample_data, method = ['SAA', 'normal']):
        sample_num = sample_data.shape[0]
        kf = KFold(n_splits = sample_num)
        eval_perform = np.zeros(sample_num)
        i = 0
        for (train_index, __) in kf.split(sample_data):
            train_data = sample_data[train_index]
            if method[0] == 'SAA':
                sol1 = self.saa(train_data)
            else:
                raise NotImplementedError
            if method[1] == 'normal':
                sol2 = self.normal(train_data)
            else:
                raise NotImplementedError

            eval_perform[i] = self.empirical_eval(sol1, train_data) - self.empirical_eval(sol2, train_data)
            i += 1

        return np.mean(eval_perform), np.std(eval_perform) * sample_num / math.sqrt(sample_num - 1)