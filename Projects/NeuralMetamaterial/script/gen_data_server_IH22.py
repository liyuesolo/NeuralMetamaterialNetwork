import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
def process(i, data):
    IH = 20
    params = data[i]
    exe_file = "../../../build/Projects/NeuralMetamaterial/NeuralMetamaterial"
    result_folder = "/" + str(i) + "/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    os.environ['OMP_THREAD_LIMIT'] = '1'
    # if os.path.exists(result_folder + "structure.vtk"):
    os.system(exe_file+" "+str(IH)+" " + result_folder + " 3 " + str(params[0]) + " " + str(params[1]) + " " + str(params[2]) + " 0")
    


param_list = []
params_range = [[0.1,0.3], [0.3, 0.7], [0.0, 0.3]]
n_sp_params = 10


for i in range(n_sp_params):
    pi = params_range[0][0] + (float(i)/float(n_sp_params))*(params_range[0][1] - params_range[0][0])
    for j in range(n_sp_params):
        pj = params_range[1][0] + (float(j)/float(n_sp_params))*(params_range[1][1] - params_range[1][0])
        for k in range(n_sp_params):
            pk = params_range[2][0] + (float(k)/float(n_sp_params))*(params_range[2][1] - params_range[2][0])
            param_list.append([pi, pj, pk])
                

Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(len(param_list)))
# Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(8))
# process(0, param_list)


