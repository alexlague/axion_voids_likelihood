# Generate tables for Pk interpolation

import numpy as np
from scipy.stats import qmc

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

def get_pk(params):

    ln10As, Omega_m, fax = params
    As = np.exp(ln10As) / 1e10
    ombh2 = 0.02233 #omb_over_omm * Omega_m * h**2
    omnuh2 = 0.0006451439
    omdh2 = Omega_m * h**2 - ombh2 -omnuh2
    omaxh2 = omdh2 * fax
    omch2 = omdh2 * (1-fax)

    proc = str(multiprocessing.current_process()._identity)
    proc = proc.replace('(', '')
    proc = proc.replace(')', '')
    proc = proc.replace(',', '')


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
    

pk_fid = get_pk([3.043, 0.31, 0.0001])[1]

def get_tk(p):
    Om, fax = p
    
    try:
        return get_pk([3.043, Om, fax])[1] / pk_fid
    except:
        return np.zeros(len(pk_fid))


for n in range(10):
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=2000)
    l_bounds = [0.1, 1e-4]
    u_bounds = [0.5, 0.5]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    
    with multiprocessing.Pool(32) as pool:
        tk_list = pool.map(get_tk, sample_scaled)
        

    np.save('tk_tables/tk_samples_'+str(n)+'.npy', sample_scaled)
    np.save('tk_tables/tk_array_'+str(n)+'.npy', np.array(tk_list))
    print(str(n)+" done!")
