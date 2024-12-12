import numpy as np
from re import sub
import subprocess
import pandas as pd
import os
from matplotlib import pylab as plt

num_samples = 20

param_dict = {
  "Tortuosity": {"uniform": [0.01, 0.1]},
  "Sorption": {"normal": [1, 2]},
  "Permeability": {"uniform": [1e-15, 1e-14]},
}

cwd = os.getcwd()
os.chdir(cwd +'/wdir')

class Sobol():
    def __init__(self, n_sample, param):
        self.n_sample = n_sample
        self.n_dim = len(param)
        self.param = param
        self.arr = np.full([31],1e-10)  # DUMMY values, TODO - find another method than np.full
    
    def run_model(self, X_1, X_2, X_3 ):
        
        #----------------prepares the model file----------------
        f1 = open(cwd + '/models/single_fracture/single_fracture.in','r')          # original file
        f1_read = f1.read()
        f1.close()
        f2 = open("multiple_run.in", "w")          # file to run model
        def replace_param(dict_replace, target):
            for old_param, new_param in list(dict_replace.items()):
                target = sub(old_param, new_param, target)
            return target
        dict_replace = {
            'TORTUOSITY_VAR': str(X_1),
            'KD_VAR': str(X_2),
            'PERM_VAR': str(X_3)
            }
        f2.write(replace_param(dict_replace, f1_read))
        f2.close() 
        
        #------------------- runs simulation------------------------
        subprocess.run(["bash", "-c", "/home/ondro/petsc/arch-linux-c-opt/bin/mpirun -n 1 /home/ondro/pflotran/src/pflotran/pflotran -input_prefix multiple_run"])
        
        #----------------- write quantity of interest to be evaluated-----------------
        conc = pd.read_csv("multiple_run-obs-0.pft", skiprows=1, header=None, sep='\s+')     
        
        btc = conc.iloc[:, 3]
        self.arr = np.vstack((self.arr, btc))
        
        conc = conc.iloc[30]
        conc = conc.loc[3]
        return conc

    def model(self, X):  
        return np.array([self.run_model(Xi[0], Xi[1], Xi[2]) for Xi in X]).reshape(-1, 1)
    
    def matrix(self):
        mat = []
        rng = np.random.default_rng()
        for i in param_dict:
            val = param_dict.get(i)
            for j in val:
                if j == 'uniform':
                    l = rng.uniform(size=(self.n_sample, 1), low=val.get(j)[0], high=val.get(j)[1])
                    mat = np.append(mat, l)
                if j == 'normal':
                    l = rng.normal(size=(self.n_sample, 1), loc=val.get(j)[0], scale=val.get(j)[1])
                    mat = np.append(mat, l)
                if j == 'triangular':
                    l = rng.triangular(size=(self.n_sample, 1), left=val.get(j)[0], mode=val.get(j)[1], right=val.get(j)[2])
                    mat = np.append(mat, l)
    
        return np.reshape(mat,(self.n_sample,self.n_dim), order='F')    
    
    def sobol_matrices(self):
        A = self.matrix()
        B = self.matrix()
        fun_A = self.model(A)
        fun_B = self.model(B)
        fun_AB = [self.model(np.column_stack((A[:, 0:i], B[:, i], A[:, i+1:]))) for i in range(self.n_dim)]
        fun_AB = np.array(fun_AB).reshape(self.n_dim, self.n_sample)
        var_Y = np.var(np.vstack([fun_A, fun_B]), axis=0)

        return fun_A, fun_B, fun_AB, var_Y
        
    def estimate_saltelli(self):
        mat = self.sobol_matrices()
        s = 1 / self.n_sample * np.sum(mat[1] * (np.subtract(mat[2], mat[0].flatten()).T), axis=0) / mat[3]
        st = 1 / (2 * self.n_sample) * np.sum(np.subtract(mat[0].flatten(), mat[2]).T ** 2, axis=0) / mat[3]
        return s, st
    
    def estimate_jansen(self):
        mat = self.sobol_matrices()
        s = (mat[3] - 1 / (2* self.n_sample)*np.sum(np.subtract(mat[1].flatten(), mat[2]).T ** 2, axis=0))/mat[3]
        st = self.estimate_saltelli()[1]
        return s, st
    
    def plot(self):
        d = pd.read_csv("multiple_run-obs-0.pft", skiprows=1, header=None, sep='\s+')
        x = d.iloc[:, 0].to_numpy()
        y = self.arr[1:self.n_sample + 1]
        plt.plot(x, y.T)
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.show()
    
class Correlation(Sobol):
    def __init__(self, n_sample, param):
        super().__init__(n_sample, param)
        
    def correlation(self):
        A = self.matrix()
        fun_A = self.model(A).flatten()
        corrcoeffs = []
        for lay, name in zip([A[:,i] for i in range(self.n_dim)], param_dict):
            c = np.min(np.corrcoef(lay, fun_A))
            corrcoeffs = np.append(corrcoeffs, c)
            
            plt.scatter(lay, fun_A)
            plt.xlabel(name)
            plt.ylabel("Concentration")
            plt.show()

        return  corrcoeffs

 
if __name__ == "__main__":
    #computeSobol = Sobol(num_samples, param_dict)
    #sal = computeSobol.estimate_saltelli()
    #print('\n' * 2)
    #print("First order indices: ", sal[0])
    #print("Total indices: ", sal[1])

    #indices = computeSobol.estimate_jansen()
    #print(indices)
    
    computeCorrelation = Correlation(num_samples, param_dict)       
    corr = computeCorrelation.correlation()
    print('\n' * 2)
    print("Correlation coefficients: ", corr)
    
    computeCorrelation.plot()


