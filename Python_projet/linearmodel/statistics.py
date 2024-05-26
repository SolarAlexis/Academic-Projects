import numpy as np
import scipy.stats as stats

import linearmodel.visualization as vis

# Statistics on a list
def mean(list):
    n = len(list)
    if n == 0:
        return 0
    else:
        return sum(list)/n

def var(list):
    n = len(list)
    if n == 0:
        return 0
    else:
        mean_list = mean(list)
        return sum([(element-mean_list)**2 for element in list])/n

def cor(list1, list2):
    n = len(list1)
    if n != len(list2):
        raise ValueError("dimensions of the list are not equal")
    else:
        mean_list1 = mean(list1)
        mean_list2 = mean(list2)
        var_list1 = var(list1)
        var_list2 = var(list2)
        temp = sum([(list1[i]-mean_list1) * (list2[i]-mean_list2) for i in range(n)]) / n

    return temp / np.sqrt(var_list1 * var_list2)
    
# Statistics on data    
def mean_data(data):
    _, m = data.shape
    res = []
    for j in range(m):
        res.append(mean(data[:,j]))
    return res

def var_data(data):
    _, m = data.shape
    res = []
    for j in range(m):
        res.append(var(data[:,j]))
    return res

def min_data(data):
    _, m = data.shape
    res = []
    for j in range(m):
        res.append(min(data[:,j]))
    return res

def max_data(data):
    _, m = data.shape
    res = []
    for j in range(m):
        res.append(max(data[:,j]))
    return res

def cor_data(data):
    _, m = data.shape
    res = []
    for i in range(m):
        temp = []
        for j in range(m):
            temp.append(cor(data[:,i], data[:,j]))
        res.append(temp)
        
    return res

class OrdinaryLeastSquares():
    def __init__(self, X, y, intercept=True,):
        new_X = np.array(X)
        new_y = np.array(y)
        if not intercept:
            self.shape = new_X.shape
            self.coeffs = np.zeros((self.shape[1],1))
            self.X = new_X
            self.y = new_y
        else:
            self.shape = (new_X.shape[0], new_X.shape[1] + 1)
            self.coeffs = np.zeros((self.shape[1],1))
            one_column = np.ones((self.shape[0],1))
            self.X = np.concatenate((new_X, one_column), axis=1)
            self.y = new_y
    
    def fit(self):
        X_T = self.X.T
        X = self.X
        self.coeffs = np.linalg.inv(X_T @ X) @ X_T @ self.y
        
    def predict(self, X):
        new_X = np.array(X)
        if self.shape[1] == new_X.shape[1]:
            return new_X @ self.coeffs
        else:
            one_column = np.ones((new_X.shape[0],1))
            new_X = np.concatenate((new_X, one_column), axis=1)
            return new_X @ self.coeffs
    
    def get_coeffs(self):
        return self.coeffs
    
    def determination_coefficient(self):
        return cor(self.X @ self.coeffs,self.y)
    
    def determination_coefficient2(self):
        y = self.y
        y_p = self.predict(self.X)
        y_m = mean(y)
        SCR = sum([(y[i] - y_p[i])**2 for i in range(self.shape[0])])
        SCT = sum([(y[i] - y_m)**2 for i in range(self.shape[0])])
        return 1 - SCR/SCT
    
    def residual_histogram(self):
        resid = self.y - self.X @ self.coeffs
        vis.histogram(resid, title="Residual histogram of the model")
        
    def residual_variance(self):
        resid = self.y - self.X @ self.coeffs
        return round(resid.T @ resid / (self.shape[0] - self.shape[1]),2)
        
    def confidance_interval(self):
        sigma_sq = self.residual_variance()
        cov_mat = sigma_sq * np.linalg.inv(self.X.T @ self.X)
        coeffs_std = np.sqrt(np.diag(cov_mat))
        alpha = 0.05
        q_t = stats.t.ppf(1 - alpha/2, df=(self.shape[0] - self.shape[1]))
        
        res = [[round(self.coeffs[i] - q_t * coeffs_std[i],2), round(self.coeffs[i] + q_t * coeffs_std[i],2)] 
               for i in range(self.shape[1])]
        return res