import subprocess
import pandas as pd
from re import sub
import openturns as ot
import openturns.viewer as viewer
from matplotlib import pylab as plt
import numpy as np

arr = np.full([31],1e-10) # DUMMY values, TODO - find another method than np.full

def run_model(X):
    X_1, X_2, X_3 = X
    
    #----------------prepares the model file----------------
    f1 = open('model/nerovnomernaVarC.in','r')          # original file
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
    series = conc.iloc[:, 3]
    global arr
    arr = np.vstack((arr, series))      
    
    conc = conc.iloc[30]
    conc = conc.loc[3]
    return [conc] 

model = ot.PythonFunction(3, 1, run_model)
model = ot.MemoizeFunction(model)
model.setInputDescription(["Tortuosity", "Retardation", "Permeability"])
model.setOutputDescription(["conc"])

#-------------------- define marginals --------------------------------
marginals = [ot.Uniform(0.01, 0.1), ot.Uniform(0, 3), ot.Uniform(1.0e-15, 1.0e-14)]
distribution = ot.ComposedDistribution(marginals)
distribution.setDescription(["Tortuosity_uni", "Retardation_uni", "Permeability_uni"])
inputNames = distribution.getDescription()

size = 100
inputDesign = distribution.getSample(size)
outputDesign = model(inputDesign)

corr_analysis = ot.CorrelationAnalysis(inputDesign, outputDesign)

pcc_indicies = corr_analysis.computePCC()

print(inputDesign)
print(outputDesign)
print()
print(pcc_indicies)

graph = ot.VisualTest.DrawPairsXY(inputDesign, outputDesign)
view = viewer.View(graph)
plt.show()
plt.close()

#----------------------------- BTCs -------------------------------
d = pd.read_csv("multiple_run-obs-0.pft", skiprows=1, header=None, sep='\s+')
x = d.iloc[:, 0].to_numpy()
y = arr[1:size+1]
plt.plot(x, y.T)
plt.show()

"""
#--------------- PCE Sobol ----------------------------------------
sizePCE = 10
inputDesignPCE = distribution.getSample(sizePCE)
outputDesignPCE = model(inputDesignPCE)

algo = ot.FunctionalChaosAlgorithm(inputDesignPCE, outputDesignPCE, distribution)
algo.run()
result = algo.getResult()
print(result.getResiduals())
print(result.getRelativeErrors())

sensitivityAnalysis = ot.FunctionalChaosSobolIndices(result)
print(sensitivityAnalysis)
firstOrder = [sensitivityAnalysis.getSobolIndex(i) for i in range(3)]
totalOrder = [sensitivityAnalysis.getSobolTotalIndex(i) for i in range(3)]
graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(inputNames, firstOrder, totalOrder)
graph.setTitle("Sobol indices by Polynomial Chaos Expansion - testcase")
view = otv.View(graph)
"""
