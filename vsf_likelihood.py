import numpy as np
import matplotlib.pyplot as plt
import zeus
#import dynesty
import multiprocessing
from mcfit import TophatVar

from scipy.interpolate import CubicSpline, griddata, bisplrep, bisplev
from scipy.special import factorial, gammaln
import os
import sys



sys.path.append('/home/alex/Documents/axions/axionHMcode_fast/')
sys.path.append('/home/alex/Documents/axions/axionHMcode_fast/axionCAMB_and_lin_PS/')
sys.path.append('/home/alex/Documents/axions/axionHMcode_fast/cosmology/')
sys.path.append('/home/alex/Documents/axions/axionHMcode_fast/axion_functions/')
sys.path.append('/home/alex/Documents/axions/axionHMcode_fast/halo_model/')

from axionCAMB_and_lin_PS import axionCAMB_wrapper 
from axionCAMB_and_lin_PS import load_cosmology  
from axionCAMB_and_lin_PS import lin_power_spectrum 

from axion_functions import axion_params

input_file_path = 'input_file.txt'
try:
    f = open(input_file_path)
except IOError:
    print("Input file not accessible, pleas check the file path")
finally:
    f.close()
    
#IMPORTANT:Change here the path to the axionCAMB executable path directory (second path in the function)
# assumes that thee axionCAMB executable is names .camb
axionCAMB_exe_path = '/home/alex/Documents/axions/axionCAMB'
if os.path.exists(axionCAMB_exe_path+'/./camb') == False:
    print("executabel axionCAMB is not in the given directory, pleas check the path")


cosmos = load_cosmology.load_cosmology_input(input_file_path)

## LOAD DATA AND COVARIANCE ##

h = 0.68
#omb_over_omm = 0.02233 / (0.1196+0.02233)
#omc_over_omm = 0.1196 / (0.1196+0.02233)
boxsize = 2048. * h
R_array = np.geomspace(5, 85, 1000)
Nbins = 10
bin_edges = np.geomspace(25, 40, Nbins+1)
bin_cens = (bin_edges[:-1]+bin_edges[1:])/2
lcdm_void_radii = np.loadtxt('data/test_new_radii_lcdm.dat')
N_data = np.histogram(lcdm_void_radii, bins=bin_edges)[0]

Euclid = True
if Euclid:
    N_data *= 4


# What aciotns to take
mcmc = True
table = False
print_info = False

## CREATE MODEL FROM AXIONCAMB ##

def get_pk(params):

    ln10As, Omega_m, fax = params
    As = np.exp(ln10As) / 1e10
    ombh2 = 0.02233 #omb_over_omm * Omega_m * h**2
    omnuh2 = 0.0006451439
    omdh2 = Omega_m * h**2 - ombh2 -omnuh2 #omc_over_omm * Omega_m * h**2
    omaxh2 = omdh2 * fax
    omch2 = omdh2 * (1-fax)

    #print(As)
    #if parallel:
    proc = str(multiprocessing.current_process()._identity)
    proc = proc.replace('(', '')
    proc = proc.replace(')', '')
    proc = proc.replace(',', '')
    #else:
    #    proc = '_'

    outdir = "/home/alex/Documents/axions/axionVoids/test_outfiles/"
    outtrans = "cosmos_transfer_out_"+ proc+ ".dat"
    axionCAMB_wrapper.axioncamb_params("paramfiles/paramfile_axionCAMB_"+ proc +".txt", output_root="",
                                       ombh2=ombh2, omch2=omch2, omaxh2=omaxh2, scalar_amp__1___=As,
                                       hubble=100*h, omnuh2=omnuh2,
                                       massless_neutrinos=2.046, massive_neutrinos=1,
                                       transfer_filename__1___=outdir+outtrans,
                                       print_info=print_info, transfer_kmax=20,)
                                       #accuracy_boost=1, l_accuracy_boost=0, l_sample_boost=0)
    axionCAMB_wrapper.run_axioncamb("paramfiles/paramfile_axionCAMB_" + proc + ".txt", 
                                    axionCAMB_exe_path, 
                                    cosmos, print_info=print_info)
    power_spec_dic_ax = lin_power_spectrum.func_power_spec_dic(outdir+ outtrans, cosmos)

    k = np.logspace(-3, np.log10(3), 200)
    Pk = CubicSpline(power_spec_dic_ax['k'], power_spec_dic_ax['power_total'])(k)
    
    return k, Pk

Pk_fid = get_pk([3.043, 0.31, 0.001])[1]


## Load pre-computed samples
samples = np.load('tk_samples_0.npy')
tk = np.load('tk_array_0.npy')
for i in range(1, 10):
    s = np.load('tk_samples_'+str(i)+'.npy')
    samples = np.concatenate((samples, s))
    tk = np.concatenate((tk, np.load('tk_array_'+str(i)+'.npy')))

# find coefficient of spline for interpolation between samples
interp = [bisplrep(samples[:,0][tk[:,0]>0], 
                   samples[:,1][tk[:,0]>0], tk[tk[:,0]>0][:,i], kx=5, ky=5) for i in range(200)]

def get_pk_interp(params):

    
    ln10As, Omega_m, fax = params
    As = np.exp(ln10As) / 1e10

    k = np.logspace(-3, np.log10(3), 200)
    #Tk = np.array([griddata(samples, tk[:,i], (Omega_m, fax), method='nearest') for i in range(200)])
    Tk = np.array([bisplev(Omega_m, fax, interp[i]) for i in range(200)])
    
    Pk = Pk_fid * Tk
    
    As_fid = 2.096805313253679e-09

    Pk *= (As / As_fid)
    
    return k, Pk

#print(get_pk_interp([3.043, 0.31, 0.001]))


def VSF_model(bias, klinear, Plinear, B_slope=0.96, B_offset=0.26):
    R, var = TophatVar(klinear, lowring=True)(np.abs(Plinear)+1e-4, extrap=True)
    
    varR = CubicSpline(R, var)
    sigmaR = CubicSpline(R, np.sqrt(var))
    #print(f"sigma8 = {sigmaR(8)}")
    sig8 = sigmaR(8)
    #    sigmaR = CubicSpline(R, np.zeros(len(R)))
    
    F_of_b = lambda b: B_slope * bias + B_offset
    b_punct = lambda beff: 0.854*beff+0.420
    
    delta_c = 1.686 # 1.686
    delta_v = -0.7 #-0.7 #-2.71 # try value
    delta_v /= b_punct(bias) # F_of_b() # divide by tracer bias
    C = 1.594
    delta_v_lin = C * (1-(1+delta_v)**-(1/C))
    D = abs(delta_v_lin) / (delta_c+abs(delta_v_lin))
    x = lambda R: D * sigmaR(R) / abs(delta_v_lin)
    f_ln_sigma = lambda R: 2*np.sum([np.exp(-(j*np.pi*x(R))**2/2)*j*np.pi*x(R)**2*np.sin(j*np.pi*D) 
                                     for j in range(1, 25)]) #50
    
    
    V = lambda R: 4/3*np.pi * R**3
    r_L = lambda R: (1+delta_v_lin)**(1/3) * R # Lagrangian radius
    
    sigma_array = sigmaR(R_array)
    #dlnsigma_dR = CubicSpline(R_array, np.gradient(np.log(sigma_array), R_array))
    dlnsigma_dR = CubicSpline(r_L(R_array), np.gradient(np.log(sigmaR(r_L(R_array))), r_L(R_array)))
    dlnsigma_dlnR = lambda R: dlnsigma_dR(R)*R
    
    model = -dlnsigma_dlnR(r_L(R_array)) * np.array([f_ln_sigma(R) for R in R_array])/V(R_array)
    
    return model, sig8


## COMPARE MODEL TO DATA ##

def loglike(params):

    #params = [ln10As, Om, fax]
    # compute pk from parameters
    lp = logprior(params)
    if np.isfinite(lp) == False:
        return -np.inf, 0
    
    #if params[-1] <= 0:
        #print("Negative f_ax")
    #    return -np.inf, 0 #-1000
    if len(params) < 3:
        params = [params[0], params[1], 0.0001]
    try:
        k, P = get_pk_interp(params)
        #print("Success!")
    except:
        #print("Pk calculation failed")
        return -np.inf, 0 #-1000
    
    # select R range for bin
    s = [(R_array > bin_edges[i]) & (R_array < bin_edges[i+1]) for i in range(Nbins)]
    model, sig8 = VSF_model(1.36355, k, P)

    # integrate model over R-range
    N_model = [np.trapz(model[s[j]], x=np.log(R_array[s[j]])) for j in range(Nbins)]
    N_model = boxsize**3 * np.array(N_model)

    Euclid = True
    if Euclid:
        N_model *= 4

    # Compare with data assuming Poisson likelihood
    #likelihood = np.prod(N_model[3:]**N_data[3:] * np.exp(-N_model[3:]) / factorial(N_data[3:]))
    #log_like = np.sum((N_data*np.log(N_model) - N_model - np.log(factorial(N_data)))[1:-1])
    log_like = np.sum((N_data*np.log(N_model) - N_model - gammaln(N_data+1))[1:-1])
    
    #if likelihood <= 0:
    #print("Non-finite error")
    #    return -np.inf, 0 #-1000

    #print(np.log(likelihood))
    return log_like, sig8

#print(loglike([3.043, 0.31, 0.001]))

def logprior(params):

    lp = 0.
    
    if params[0] <= 2. or params[0] >= 4.:
        lp = -np.inf
    elif params[1] <= 0.1 or params[1] >= 0.5:
        lp = -np.inf
    if len(params) == 3:
        if params[2] <= 0 or params[2] >= 0.5:
            lp = -np.inf
    
    return lp


## RUN MCMC ##
# Define our uniform prior.
'''
def ptform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""
    u[0] = -0.04 * (2. * u[0] - 1.) + 3.043
    u[1] = -0.004 * (2. * u[1] - 1.) + 0.31
    u[2] = 0.05 * u[2]
    return u


# "Static" nested sampling.
ndim = 3
from multiprocessing import Pool

with Pool(32) as pool:
    sampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim, pool=pool, bound='single', nlive=1000, queue_size=32)
    sampler.run_nested(checkpoint_file='dynesty.save')
    #sampler = dynesty.NestedSampler.restore('dynesty.save', pool=pool)
    #sampler.run_nested(checkpoint_file='dynesty.save')

res = sampler.results
samples, weights = res.samples, res.importance_weights()
np.savetxt('samples.txt', samples)
np.savetxt('weights.txt', weights)

'''

if mcmc:
    nsteps, nwalkers, ndim = 2000, 20, 2
    rng = np.random.default_rng()
    mean = np.array([3.043, 0.31, 0.01])[:ndim]
    cov = np.diag(np.array([0.01, 0.01, 0.05])[:ndim]**2)
    start = rng.multivariate_normal(mean, cov, nwalkers)
    start = np.abs(start)

    cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=100, dact=0.01, nact=50, discard=0.5)
    cb1 = zeus.callbacks.SplitRCallback(ncheck=100, epsilon=0.01, nsplits=2, discard=0.5)
    cb2 = zeus.callbacks.MinIterCallback(nmin=500)
    
    
    from multiprocessing import Pool
    with Pool(20) as pool:
        sampler = zeus.EnsembleSampler(nwalkers, ndim, loglike, pool=pool, blobs_dtype=[('sigma8', float)])
        sampler.run_mcmc(start, nsteps, callbacks=[cb0, cb1, cb2])

    chain = sampler.get_chain(flat=True)
    chain_sig8 = sampler.get_blobs(flat=True)
    np.save('test_lcdm_chain_Euclid.npy', chain)
    np.save('test_lcdm_chain_sig8_Euclid.npy', chain_sig8)

## GENERATE LOOKUP TABLE ##
if table:

    pk_fid = get_pk([3.043, 0.31, 0.0001])[1]
    
    def get_tk(p):
        Om, fax = p
        #fax = 10**fax
        try:
            return get_pk([3.043, Om, fax])[1] / pk_fid
        except:
            return np.zeros(len(pk_fid))


    from scipy.stats import qmc

    for n in range(10):
        sampler = qmc.LatinHypercube(d=2)
        sample = sampler.random(n=2000)
        l_bounds = [0.1, 1e-4]
        u_bounds = [0.5, 0.5]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

        with multiprocessing.Pool(32) as pool:
            tk_list = pool.map(get_tk, sample_scaled)


        np.save('tk_samples_'+str(n)+'.npy', sample_scaled)
        np.save('tk_array_'+str(n)+'.npy', np.array(tk_list))
        print(str(n)+" done!")
