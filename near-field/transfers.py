import math
import numpy as np

def preprocess(a_v, v_dh, E_b):
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
    #a_v = 1e-4 # volume aperture of the fracture
    #v_dh = 2 # flow velocity of groundwater in the fracture
    #E_b = 0.5 # porosity of the buffer
    V_b = 2600 # volume of the buffer 
    S_c = 0.5 # buffer above waste package

    # Buffer to tunnel
    q_bt = (math.pi * (r_dh**2) * D_eb) / S_c
    transfer_bt = q_bt / E_b * V_b

    # Buffer to fracture
    q_bf = 2 * math.pi * r_dh * a_v * math.sqrt((4 * D_w * v_dh) / ((math.pi**2) * r_dh ))
    transfer_bf = q_bf / E_b * V_b
    #transfer_bf = 0.2 /V_b * E_b

    import pandas as pd

    data_dict = {'canister': {'canister': 0, 'bentonite': transfer_c, 'fracture': 0},
                 'bentonite': {'canister': 0, 'bentonite': 0, 'fracture': transfer_bf},
                 'fracture': {'canister': 0, 'bentonite': 0, 'fracture': 0}}
    matrix = pd.DataFrame(data_dict).T

    matrix.to_csv('transfers.csv')
    return matrix

if __name__ == "__main__":
    preprocess(1e-4, 2, 0.5)