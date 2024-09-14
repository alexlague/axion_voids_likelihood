# Contains the voids size function Poisson likelihood

import numpy as np
from scipy.special import gammaln
from model import VSF_model, get_pk_interp, variance

h = 0.6737
R_array = np.geomspace(5, 85, 1_000)
Nbins = 10
bin_edges = np.geomspace(25, 40, Nbins+1)
bin_cens = (bin_edges[:-1]+bin_edges[1:])/2
#void_radii = np.loadtxt('data/test_new_radii_lcdm.dat')
void_radii = np.loadtxt('data/test_new_radii_m25.dat')
N_data = np.histogram(void_radii, bins=bin_edges)[0]
boxsize = 2048*h

Euclid = False
DESI = True
if Euclid:
    factor = 5
elif DESI:
    factor = 8
else:
    factor = 1

N_data *= factor


def logprior(params):

    lp = 0.

    if params[0] <= 2. or params[0] >= 4.:
        lp = -np.inf
    elif params[1] <= 0.1 or params[1] >= 0.5:
        lp = -np.inf
    # if axions present impose Gaussian prior on params 0, 1
    #if len(params) == 3:
    As = np.exp(params[0]) # removing /1e10
    As_fid = np.exp(3.043)
    lp += - (As - As_fid)**2 / 2 / 0.3**2
    lp += - (params[1] - 0.31)**2 / 2 / 0.002**2
    if params[2] <= 0 or params[2] >= 0.5:
        lp = -np.inf

    return lp


def loglike(params, b=1.26, B_offset=0.420, use_prior=True):

    # Prior
    if use_prior:
        lp = logprior(params)
    else:
        lp = 0.
    if np.isfinite(lp) == False:
        return -np.inf, 0

    if len(params) < 3:
        params = [params[0], params[1], 0.0001]
    try:
        k, P = get_pk_interp(params)

    except:

        return -np.inf, 0 

    # select R range for bin
    s = [(R_array > bin_edges[i]) & (R_array < bin_edges[i+1]) for i in range(Nbins)]
    R, var = variance(k, P)
    model, sig8 = VSF_model(b, R, var, B_offset=B_offset) # 1.3655 for lcdm data

    # integrate model over R-range
    N_model = [np.trapz(model[s[j]], x=np.log(R_array[s[j]])) for j in range(Nbins)]
    N_model = boxsize**3 * np.array(N_model)

    N_model *= factor

    # Compare with data assuming Poisson likelihood                                                                                                                    

    log_like = np.sum((N_data*np.log(N_model) - N_model - gammaln(N_data+1))[3:-1]) #[1:-1])
    log_like += lp # add prior if finite
    return log_like, sig8

# test
print(loglike([3.043, 0.31, 0.0001]))
