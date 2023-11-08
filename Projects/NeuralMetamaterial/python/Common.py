import numpy as np
import os
from model import *

def CauchyToGreen(cauchy):
    if cauchy < 0:
        return cauchy - 0.5 * cauchy * cauchy
    else:
        return cauchy + 0.5 * cauchy * cauchy
    
def loadModel(IH, use_double = False):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    bounds = []
    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti_default = np.array([0.104512, 0.65])
    elif IH == 50:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti_default = np.array([0.23076, 0.5])
    elif IH == 67:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1]) 
        ti_default = np.array([0.2308, 0.8696])
    elif IH == 22:
        n_tiling_params = 3
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1]) 
        bounds.append([0.0, 0.3]) 
        ti_default = np.array([0.2308, 0.5, 0.2253])
    elif IH == 29:
        n_tiling_params = 1
        bounds.append([0.005, 1.0])
        ti_default = np.array([0.3669])
    elif IH == 28:
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        ti_default = np.array([0.4528, 0.5])
    elif IH == 1:
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        ti_default = np.array([0.1224, 0.5, 0.1434, 0.625])
    
    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)
    if use_double:
        model_name += "double"
    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    if use_double:
        model = buildNMNModel(n_tiling_params, tf.float64)
    else:
        model = buildNMNModel(n_tiling_params, tf.float32)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    return model, n_tiling_params, ti_default, bounds
