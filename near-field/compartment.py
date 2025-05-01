import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

import transfers

final_time = 100000
n_steps = 100000

#transfers.preprocess(1e-4, 2, 0.5)

#---------------------------------- Engine ------------------------------------
# Nuclide struct
class Nuclide:
    def __init__(self, name, initial_amount, decay_constant, transfer_to=None, decay_to=None, retardation = 1):
        self.name = name
        self.initial_amount = initial_amount
        self.decay_constant = decay_constant
        self.transfer_to = transfer_to or []  
        self.decay_to = decay_to  or [] 
        self.retardation = retardation

def nearfield_ode(t, y, nuclides):
    dydt = [0.0] * len(nuclides)
    for i, nuclide in enumerate(nuclides):
        # Radioactive decay
        for target_branch_index, decay_rate in nuclide.decay_to:
            dydt[target_branch_index] += decay_rate * y[i] * nuclide.decay_constant
            dydt[i] -= decay_rate * y[i] * nuclide.decay_constant #Loss from original compartment
            
        # Transfers to other compartments
        for target_transfer_index, transfer_rate in nuclide.transfer_to:
            dydt[target_transfer_index] += (transfer_rate / nuclide.retardation) * y[i]
            dydt[i] -= (transfer_rate / nuclide.retardation) * y[i] #Loss from original compartment       
    return dydt

t_span = (0, final_time)
t_eval = np.linspace(t_span[0], t_span[1], n_steps)


#------------------------ Process inputs --------------------------------------              
df_nuclides = pd.read_csv('nuclides.csv')
df_transfers = pd.read_csv('transfers.csv')
df_source = pd.read_csv('source.csv')
df_kd = pd.read_csv('kd.csv')
        
def process_1D(df_input_data):
    input_data = []
    for col in df_input_data.columns[1:]:
        for value in df_input_data[col]:
            input_data.append(value)
    return(input_data)

def process_2D(df_nuclides, df_transfers):
    nuclides_aux = []
    for index_tr, row_tr in df_transfers.iterrows():
        for index_nuc, row_nuc in df_nuclides.iterrows():
            transfer = [((index-1)*(len(df_nuclides))+index_nuc, value) for index, value in enumerate(row_tr) if value !=0]
            transfer = transfer[1:]
            name = f"{row_tr.iloc[0]}_{row_nuc.iloc[0]}"
            initial_amount = float(0.)
            #if index_tr < len(df_transfers)-1:
            #    decay = float(row_nuc.iloc[1])
            #else:
             #   decay = 0
            decay = float(row_nuc.iloc[1])
            branch_ratio = row_nuc[2:]
            branch = [(index+index_tr*(len(df_nuclides)), value) for index, value in enumerate(branch_ratio) if value !=0]
            
            nuclide = [name, initial_amount, decay, transfer, branch]
            nuclides_aux.append(nuclide)
    return nuclides_aux

nuclides=[]        
for nuc, init, kd in zip(process_2D(df_nuclides, df_transfers), process_1D(df_source), process_1D(df_kd)):
    nuc[1] = init
    nuc.append(kd)
    nuclide = Nuclide(nuc[0], nuc[1], nuc[2], nuc[3], nuc[4])
    nuclides.append(nuclide)
    
y0 = [nuclide.initial_amount for nuclide in nuclides]
solution = solve_ivp(nearfield_ode, t_span, y0, args=(nuclides,), t_eval=t_eval, method="BDF")

#------------------------- Plot the results -----------------------------------
plt.plot(solution.t, solution.y[0], label="nuclide A in canister")
plt.plot(solution.t, solution.y[1], label="nuclide B in canister")
plt.plot(solution.t, solution.y[2], label="nuclide A in buffer")
plt.plot(solution.t, solution.y[3], label="nuclide B in buffer")
plt.plot(solution.t, solution.y[4], label="nuclide A in fracture")
plt.plot(solution.t, solution.y[5], label="nuclide B in fracture")
plt.plot(solution.t, solution.y[6], label="nuclide A in outflow")
plt.plot(solution.t, solution.y[7], label="nuclide B in outflow")

plt.xlabel("Time")
plt.ylabel("mol")
plt.title("Near-field model")
plt.legend()
plt.grid(True)
plt.show()

#----------------------- Generate pflotran files ------------------------------
def look_up_table(solution):
    table = [solution[i] - solution[i-1] for i in range(len(solution))]
    table[0] = 0
    return table

A = solution.y[4].tolist()
A = look_up_table(A)

B = solution.y[5].tolist()
B = look_up_table(B)

plt.plot(solution.t, solution.y[4], label="nuclide B source")
plt.plot(solution.t, solution.y[5], label="nuclide A source")
plt.xlabel("Time")
plt.ylabel("mol")
plt.legend()
plt.grid(True)
plt.show()

f1 = open('constraints.txt', 'w')
f1.write('''CONSTRAINT initial
  CONCENTRATIONS
    I129     1.d-20      T
  /
END''')

f2 = open('conditions.txt', 'w')
f2.write('''
  TRANSPORT_CONDITION initial
    TYPE ZERO_GRADIENT
    CONSTRAINT_LIST
      0.d0 initial
  /
END

  TRANSPORT_CONDITION inlet
    TYPE DIRICHLET_ZERO_GRADIENT
    CONSTRAINT_LIST''')
f2.close()

f1 = open('constraints.txt', "a")
f2 = open('conditions.txt', 'a')
for i, val in enumerate(A):
    f1.write('''CONSTRAINT inlet''' + str(i) +'''
  CONCENTRATION
    I129   '''  + str(val) + '''    T  
  /
END

''')
    f2.write('''
    ''' + str(i*(final_time/n_steps)) + ''' inlet''' + str(i) + ''' ''')
f2.close()             
f1.close()

f2 = open('conditions.txt', 'a')
f2.write(''' 
    /
END''')
f2.close()
