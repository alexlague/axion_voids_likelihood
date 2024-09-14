# Calibrate the F_of_b function to recover
# unbiased parameters after mcmc

import numpy as np
from scipy.optimize import minimize
from likelihood import loglike


def to_min(B_offset):

    # want max likelihood
    #out = -loglike([3.043, 0.31, 0.0001], b=1.51, B_offset=B_offset, use_prior=False)[0]
    #out = -loglike([3.043, 0.31, 0.0001], b=1.26, B_offset=B_offset, use_prior=False)[0] 
    out = -loglike([3.043, 0.31, 0.1], b=1.51, B_offset=B_offset, use_prior=True)[0] 
    
    return out

result = minimize(to_min, x0=[.5], method="Nelder-Mead")['x']

print(result)
