import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
def process(i, data):
    IH = -1
    params = data[i]
    exe_file = "../../../build/Projects/NeuralMetamaterial/NeuralMetamaterial"
    result_folder = "/" + str(i) + "/"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    os.system(exe_file+" "+str(IH)+" " + result_folder + " " + str(params[0]))
    
param_list = []
params_range = [[-1.0, 1.0]]
n_sp_params = 400

for i in range(n_sp_params + 1):
    pi = params_range[0][0] + (float(i)/float(n_sp_params))*(params_range[0][1] - params_range[0][0])
    param_list.append([pi])

Parallel(n_jobs=8)(delayed(process)(i, param_list) for i in range(len(param_list)))



