import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
def process(i, data):
    IH = 60
    params = data[i]
    exe_file = "../../../build/Projects/NeuralMetamaterial/NeuralMetamaterial"
    result_folder = "/" + str(i) + "/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    os.environ['OMP_THREAD_LIMIT'] = '1'
    # if os.path.exists(result_folder + "structure.vtk"):
    os.system(exe_file+" "+str(IH)+" " + result_folder + " 2 " + str(params[0]) + " " + str(params[1]) + " 0")
    


param_list = []
params_range = [[0.1,0.3], [0.6, 1.1]]
n_sp_params = 20


for i in range(n_sp_params):
    pi = params_range[0][0] + (float(i)/float(n_sp_params))*(params_range[0][1] - params_range[0][0])
    for j in range(n_sp_params):
        pj = params_range[1][0] + (float(j)/float(n_sp_params))*(params_range[1][1] - params_range[1][0])
        param_list.append([pi, pj])
                

Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(len(param_list)))
# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(8))
# process(0, param_list)


