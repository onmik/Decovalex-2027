import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Transfer constant as in "Poteri (2013): Simplifying solute transport modelling of the geological multi-barrier disposal system" http://lib.tkk.fi/Diss/2013/isbn9789513880989/isbn9789513880989.pdf

# For the canister
r_h = 10 # the radius of the hole
r_u = 10 # the outer radius of the hole in the side of the buffer
D_w = 1e-9 # the molecular diffusion coefficient in free water
D_eb = 1e-10 # the effective molecular diffusion coefficient in the buffer
l_c = 0.01 # the thickness of the canister wall
V_c = 500 # the volume of the canister

# Canister to buffer
q_ch = (math.pi * r_h**2 * D_w) / l_c # the equivalent flow rate through the hole
q_hm = ((r_h * r_u) / (r_h + r_u)) * 2 * math.pi * D_eb # the equivalent flow rate on the bentonite side of the hole
q_c = (q_ch * q_hm) / (q_ch + q_hm) # the equivalent flow rate through the hole and bentonite side of the hole
transfer_c = q_c/V_c # the decay constant of the solute for the mass transfer from canister to buffer
transfer_c = 5e-3/V_c
transfer_c = 1

# For the buffer
r_dh = 1.750 / 2 # radius of the deposition hole
a_v = 1e-4 # volume aperture of the fracture
v_dh = 2 # flow velocity of groundwater in the fracture
E_b = 0.5 # porosity of the buffer
V_b = 2600 # volume of the buffer 
S_c = 0.5 # buffer above waste package

# Buffer to tunnel
q_bt = (math.pi * (r_dh**2) * D_eb) / S_c
transfer_bt = q_bt / E_b * V_b

# Buffer to fracture
q_bf = 2 * math.pi * r_dh * a_v * math.sqrt((4 * D_w * v_dh) / ((math.pi**2) * r_dh ))
transfer_bf = q_bf / E_b * V_b
#transfer_bf = 0.2 /V_b * E_b

# Nuclide struct
class Nuclide:
    def __init__(self, name, initial_amount, decay_constant, transfer_to=None, decay_to=None, retardation = 1):
        self.name = name
        self.initial_amount = initial_amount
        self.decay_constant = decay_constant
        self.transfer_to = transfer_to or []  # List of (target_index, transfer_rate) tuples
        self.decay_to = decay_to   # (target_index, branching_ratio) or None if stable
        self.retardation = retardation

def nearfield_ode(t, y, nuclides):
    #ODE system for biosphere compartment model with decay chain.
    dydt = [0.0] * len(nuclides)

    for i, nuclide in enumerate(nuclides):
        dydt[i] -= nuclide.decay_constant * y[i]  # Radioactive decay

        #Production from the parent nuclide.
        for j, parent_nuclide in enumerate(nuclides):
            if parent_nuclide.decay_to:
                if parent_nuclide.decay_to[0] == i:
                    dydt[i] += parent_nuclide.decay_to[1] * parent_nuclide.decay_constant * y[j]

        # Transfers to other compartments
        for target_transfer_index, transfer_rate in nuclide.transfer_to:
            dydt[target_transfer_index] += (transfer_rate / nuclide.retardation) * y[i]
            dydt[i] -= (transfer_rate / nuclide.retardation) * y[i] #Loss from original compartment

    return dydt

# Example biosphere model: buffer -> fracture, with a decay chain in the buffer
nuclide_A_canister = Nuclide("canister_A", 1.0, 5e-4,[(2, transfer_c)], (1, 1))
nuclide_B_canister = Nuclide("canister_B", 0., 0.,[(3, transfer_c)], () )
nuclide_A_buffer = Nuclide("buffer_A", 0.0, 5e-4, [(4, transfer_bf)], (3, 1), 1) # buffer A -> fracture A, buffer A -> buffer B
nuclide_B_buffer = Nuclide("buffer_B", 0.0, 0., [(5, transfer_bf)], (), 1) #buffer B decay.
nuclide_A_fracture = Nuclide("fracture_A", 0.0, 0.0, [], ()) #fracture A decay. 
nuclide_B_fracture = Nuclide("fracture_B", 0.0, 0.0) #fracture B decay.

nuclides = [nuclide_A_canister, nuclide_B_canister, nuclide_A_buffer, nuclide_B_buffer, nuclide_A_fracture, nuclide_B_fracture]

# Initial conditions
y0 = [nuclide.initial_amount for nuclide in nuclides]

# Time span and evaluation points
final_time = 60000
n_steps = 10000
t_span = (0, final_time)
t_eval = np.linspace(t_span[0], t_span[1], n_steps)

# Solve the ODEs
solution = solve_ivp(nearfield_ode, t_span, y0, args=(nuclides,), t_eval=t_eval, method="BDF") # Implicit method for stiff problem is preffered, also works
#solution = solve_ivp(nearfield_ode, t_span, y0, args=(nuclides,), method="BDF") # automatic time steps

# Plot the results
plt.plot(solution.t, solution.y[0], label="nuclide A in canister")
plt.plot(solution.t, solution.y[1], label="nuclide B in canister")
plt.plot(solution.t, solution.y[2], label="nuclide A in buffer")
plt.plot(solution.t, solution.y[3], label="nuclide B in buffer")
plt.plot(solution.t, solution.y[4], label="nuclide A in fracture")
plt.plot(solution.t, solution.y[5], label="nuclide B in fracture")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Near-field model")
plt.legend()
plt.grid(True)
plt.show()


def look_up_table(solution):
    table = [solution[i] - solution[i-1] for i in range(len(solution))]
    table[0] = 0
    return table

A = solution.y[4].tolist()
A = look_up_table(A)

B = solution.y[5].tolist()
B = look_up_table(B)

plt.plot(solution.t, B, label="nuclide B source")
plt.plot(solution.t, A, label="nuclide A source")
plt.xlabel("Time")
plt.ylabel("Amount")
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






