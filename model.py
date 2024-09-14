##
## Model for the void size function from
## interpolated matter power spectrum
##

import numpy as np
from scipy.interpolate import CubicSpline, bisplrep, bisplev
from mcfit import TophatVar
import multiprocessing
import time
import sys
import os

print_info = False

# for axionHMCode
hmcode_path = '/home/alex/Documents/axions/axionHMcode_fast/'
axionCAMB_exe_path = '/home/alex/Documents/axions/axionCAMB'
outdir = "/home/alex/Documents/axions/axionVoids/test_outfiles/" # directory where the output of axioncamb is stored
input_file_path = 'input_file.txt'
sys.path.append(hmcode_path)
sys.path.append(hmcode_path + 'axionCAMB_and_lin_PS/')
sys.path.append(hmcode_path + 'cosmology/')
sys.path.append(hmcode_path + 'axion_functions/')
sys.path.append(hmcode_path + 'halo_model/')

from axionCAMB_and_lin_PS import axionCAMB_wrapper
from axionCAMB_and_lin_PS import load_cosmology
from axionCAMB_and_lin_PS import lin_power_spectrum
from axion_functions import axion_params

# check input file
try:
    f = open(input_file_path)
except IOError:
    print("Input file not accessible, please check the file path")
finally:
    f.close()

# check axionCAMB
if os.path.exists(axionCAMB_exe_path+'/./camb') == False:
    print("executable axionCAMB is not in the given directory, please check the path")
    
def load_pk_table():
    """
    Load lookup tables and return interpolation coefficients
    """
    samples = np.load('tk_tables/tk_samples_0.npy')
    tk = np.load('tk_tables/tk_array_0.npy')
    for i in range(1, 10):
        s = np.load('tk_tables/tk_samples_'+str(i)+'.npy')
        samples = np.concatenate((samples, s))
        tk = np.concatenate((tk, np.load('tk_tables/tk_array_'+str(i)+'.npy')))

    # Find coefficient of spline for interpolation between samples
    interp = [bisplrep(samples[:,0][tk[:,0]>0],
                       samples[:,1][tk[:,0]>0], tk[tk[:,0]>0][:,i], kx=5, ky=5) for i in range(200)]

    return interp
    
def get_pk(params):
    """
    Generate matter power spectrum from axionCAMB interface (from axionHMode)
    """

    # fiducial and varied parameters
    ln10As, Omega_m, fax = params
    As = np.exp(ln10As) / 1e10
    h = 0.6737
    ombh2 = 0.02233 
    omnuh2 = 0.0006451439
    omdh2 = Omega_m * h**2 - ombh2 -omnuh2
    omaxh2 = omdh2 * fax
    omch2 = omdh2 * (1-fax)

    # this allows writing from multiple threads
    proc = str(multiprocessing.current_process()._identity)
    proc = proc.replace('(', '')
    proc = proc.replace(')', '')
    proc = proc.replace(',', '')

    outtrans = "cosmos_transfer_out_"+ proc+ ".dat"
    axionCAMB_wrapper.axioncamb_params("paramfiles/paramfile_axionCAMB_"+ proc +".txt", output_root="",
                                       ombh2=ombh2, omch2=omch2, omaxh2=omaxh2, scalar_amp__1___=As,
                                       hubble=100*h, omnuh2=omnuh2,
                                       massless_neutrinos=2.046, massive_neutrinos=1,
                                       transfer_filename__1___=outdir+outtrans,
                                       print_info=print_info, transfer_kmax=20,)
                                       
    axionCAMB_wrapper.run_axioncamb("paramfiles/paramfile_axionCAMB_" + proc + ".txt",
                                    axionCAMB_exe_path,
                                    cosmos, print_info=print_info)
    power_spec_dic_ax = lin_power_spectrum.func_power_spec_dic(outdir+ outtrans, cosmos)

    k = np.logspace(-3, np.log10(3), 200)
    Pk = CubicSpline(power_spec_dic_ax['k'], power_spec_dic_ax['power_total'])(k)

    return k, Pk



def get_pk_interp(params):
    """
    Get the interpolated power spectrum from cosmological params and lookup table
    """

    ln10As, Omega_m, fax = params
    As = np.exp(ln10As) / 1e10

    k = np.logspace(-3, np.log10(3), 200)

    Tk = np.zeros(k.shape)
    for i in range(len(k)):
        Tk[i] = bisplev(Omega_m, fax, interp[i])

    Pk = Pk_fid * Tk

    As_fid = 2.096805313253679e-09

    Pk *= (As / As_fid)

    return k, Pk

def variance(klinear, Plinear):
    """
    
    """
    R, var = TophatVar(klinear, lowring=True)(np.abs(Plinear)+1e-4, extrap=True)
    return R, var


def VSF_model(bias, R, var, B_slope=0.854, B_offset=0.420):
    """
    """
    R_array = np.geomspace(5, 85, 1000)
    
    sigmaR = CubicSpline(R, np.sqrt(var))
    sig8 = np.interp([8.], R, np.sqrt(var))
    sigma_array = np.interp(R_array, R, np.sqrt(var))

    b_punct = lambda beff: B_slope * beff + B_offset #0.854*beff+0.420
    delta_c = 1.686
    delta_v = -0.7 
    delta_v /= b_punct(bias) # F_of_b() # divide by tracer bias
    C = 1.594
    delta_v_lin = C * (1-(1+delta_v)**-(1/C))
    D = abs(delta_v_lin) / (delta_c+abs(delta_v_lin))

    x = lambda R: D * sigmaR(R) / abs(delta_v_lin)
    f_ln_sigma = lambda R: 2*np.sum([np.exp(-(j*np.pi*x(R))**2/2)*j*np.pi*x(R)**2*np.sin(j*np.pi*D)
                                     for j in range(1, 25)]) #50
    x_array = D * sigma_array / abs(delta_v_lin)
    f_ln_sigma_array = np.zeros(R_array.shape)
    for j in range(1, 25):
        f_ln_sigma_array += 2 * np.exp(-(j*np.pi*x_array)**2/2)*j*np.pi*x_array**2*np.sin(j*np.pi*D)

    V = lambda R: 4/3*np.pi * R**3
    r_L = lambda R: (1+delta_v_lin)**(1/3) * R # Lagrangian radius
    V_array = 4/3*np.pi*R_array**3
    r_L_array = (1+delta_v_lin)**(1/3) * R_array

    dlnsigma_dR = CubicSpline(r_L(R_array), np.gradient(np.log(sigmaR(r_L(R_array))), r_L(R_array)))
    dlnsigma_dlnR = lambda R: dlnsigma_dR(R)*R

    model = -dlnsigma_dlnR(r_L(R_array)) * f_ln_sigma_array / V_array
    
    return model, sig8


## main function ##
cosmos = load_cosmology.load_cosmology_input(input_file_path)
interp = load_pk_table()
Pk_fid = get_pk([3.043, 0.31, 0.001])[1]
start = time.process_time()
get_pk_interp([3.043, 0.31, 0.0001])
print("Time to generate Pk from interpolation: ", time.process_time() - start)
