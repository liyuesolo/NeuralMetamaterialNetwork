import os
from functools import cmp_to_key
from joblib import Parallel, delayed

from scipy.optimize import BFGS
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from requests import options
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import math
import numpy as np
import tensorflow as tf
from model import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import time
from Derivatives import *
from Optimization import *
from PropertyModifier import *
from Common import *

use_double = True

if use_double:
    tf.keras.backend.set_floatx("float64")
else:
    tf.keras.backend.set_floatx("float32")

def CauchyToGreen(cauchy):
    if cauchy < 0:
        return cauchy - 0.5 * cauchy * cauchy
    else:
        return cauchy + 0.5 * cauchy * cauchy

# check analytical derivative NeoHookean
@tf.function
def psiGradHessNH(strain, data_type = tf.float32):
    lambda_tf = 26.0 * 0.48 / (1.0 + 0.48) / (1.0 - 2.0 * 0.48)
    mu_tf = 26.0 / 2.0 / (1.0 + 0.48)
    
    batch_dim = strain.shape[0]
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(strain)
        with tf.GradientTape() as tape:
            tape.watch(strain)
            
            strain_xx = tf.gather(strain, [0], axis = 1)
            strain_yy = tf.gather(strain, [1], axis = 1)
            
            strain_xy = tf.constant(0.5, dtype=data_type) * tf.gather(strain, [2], axis = 1)
            strain_vec_reorder = tf.concat((strain_xx, strain_xy, strain_xy, strain_yy), axis=1)
            
            strain_tensor = tf.reshape(strain_vec_reorder, (batch_dim, 2, 2))    
                        
            righCauchy = tf.constant(2.0, dtype=data_type) * strain_tensor + tf.eye(2, batch_shape=[batch_dim], dtype=data_type)
            
            J = tf.math.sqrt(tf.linalg.det(righCauchy))
            
            I1 = tf.linalg.trace(righCauchy)
            C1 = tf.constant(0.5 * mu_tf, dtype=data_type)
            D1 = tf.constant(lambda_tf * 0.5, dtype=data_type)
            lnJ = tf.math.log(J)
            psi = C1 * (I1 - tf.constant(2.0, dtype=data_type) - tf.constant(2.0, dtype=data_type) * lnJ) + D1 * (lnJ*lnJ)
            
            stress = tape.gradient(psi, strain)
            # print(stress)
            # exit(0)
    C = tape_outer.batch_jacobian(stress, strain)
    del tape
    del tape_outer
    return psi, stress, C



def toPolarData(half):
    full = half
    n_sp_theta = len(half)
    for i in range(n_sp_theta):
        full = np.append(full, full[i])
    full = np.append(full, full[0])
    return full



def generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti, model):
    uniaxial_strain = []
    for theta in thetas:
        uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, ti, False)
        uniaxial_strain.append(uni_strain)
    # print(uniaxial_strain)
    # exit(0)
    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    uniaxial_strain = np.reshape(uniaxial_strain, (batch_dim, 3))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    stiffness = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()
    return stiffness

def stiffnessOptimizationSA(IH, plot_sim = False, plot_GT = False):
    
    bounds = []

    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    strain = 0.02
    current_dir = os.path.dirname(os.path.realpath(__file__))
    idx = np.arange(0, len(thetas), 5)

    if IH == 21:
        strain = 0.02
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti = np.array([0.18, 0.7])
        ti_target = np.array([0.1045, 0.65])
        sample_idx = [2, 7, -1]
        theta = 0.0

    elif IH == 50:
        strain = 0.05
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti = np.array([0.2308, 0.5])
        ti_target = np.array([0.2903, 0.6714])
        
    elif IH == 67:
        strain = 0.1
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        ti = np.array([0.18, 0.68])
        ti_target = np.array([0.25, 0.85])
    elif IH == 22:
        strain = 0.02
        n_tiling_params = 3
        bounds.append([0.1, 0.3])
        bounds.append([0.3, 0.7]) 
        bounds.append([0.0, 0.3])
        ti_target = np.array([0.14, 0.6, 0.3])
        ti = np.array([0.12, 0.5, 0.22])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 29:
        strain = 0.2
        n_tiling_params = 1
        bounds.append([0.005, 1.0])
        ti_target = np.array([0.09])
        ti = np.array([0.2])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 28:
        strain = 0.02
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        ti_target = np.array([0.4, 0.6])
        ti = np.array([0.2, 0.6])
        # ti_target = np.array([0.6, 0.6])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 1:
        strain = 0.05
        # strain = CauchyToGreen(strain)
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        # test 1
        ti = np.array([0.1224, 0.5, 0.1434, 0.625])
        # ti_target = np.array([0.1224, 0.6, 0.13, 0.625])
        # test 2
        # ti = np.array([0.1224, 0.6, 0.13, 0.625])
        # ti_target = np.array([0.13, 0.4998, 0.11, 0.6114])
        # test 3
        # ti = np.array([0.13, 0.4998, 0.11, 0.6114])
        # ti_target = np.array([0.1224, 0.5, 0.1087, 0.5541])
        # test 4
        # ti = np.array([0.1224, 0.5, 0.1087, 0.55408])
        # ti_target = np.array([0.1224, 0.5, 0.1434, 0.625])
        # test 5
        # ti = np.array([0.1692, 0.4223, 0.0635, 0.6888])
        # ti_target = np.array([0.1224, 0.4724, 0.12, 0.625])
        # test 6
        # ti = np.array([0.1224, 0.4724, 0.12, 0.625])
        # ti_target = np.array([0.16, 0.5, 0.12, 0.55])
        # test 7
        # ti = np.array([0.16, 0.5, 0.12, 0.55])
        # ti_target = np.array([0.22, 0.6, 0.08, 0.6])
        # test 8
        # ti = np.array([0.18710856, 0.58457689, 0.10264114, 0.74953785])
        # ti_target = np.array([0.1692, 0.4223, 0.0635, 0.6888])
        # 0.19396568 0.46893408 0.06722148 0.75063715

        # ti = np.array([0.2434, 0.4494, 0.0494, 0.625])
        # ti = np.array([0.16, 0.5, 0.12, 0.55])
        ti_target = np.array([0.26, 0.75,  0.15,  0.58])
        # ti_target = np.array([0.1949, 0.6434, 0.1403, 0.6858])
        idx = np.arange(0, len(thetas), 5)

    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)

    model_name += "double"
    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    if use_double:
        model = buildNMNModel(n_tiling_params, tf.float64)
    else:
        model = buildNMNModel(n_tiling_params, tf.float32)
    model.load_weights(save_path + "IH" + model_name + '.tf')


    # uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    # print(uniaxial_strain)
    stiffness = generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti, model)

    stiffness_targets = generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti_target, model)
    
    
    if IH == 21:
        mean = np.mean(stiffness_targets)
        stiffness_targets = np.full((len(stiffness_targets), ), mean)
    
    sample_points_theta = thetas[idx]
    batch_dim = len(thetas)
    stiffness_targets_sub = stiffness_targets[idx]
    base_folder = "../results/stiffness/"
    def objAndGradient(x):
        _uniaxial_strain = []
        dqdp = []
        for theta in thetas:
            uni_strain, dqidpi = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, x, False)
            _uniaxial_strain.append(uni_strain)
            dqdp.append(dqidpi)
        

        ti_TF = tf.convert_to_tensor(x)

        uniaxial_strain_TF = tf.convert_to_tensor(_uniaxial_strain)
        stiffness_current, stiffness_grad, dOdE = objGradStiffness( 
                                            ti_TF, uniaxial_strain_TF, 
                                            tf.convert_to_tensor(thetas), 
                                            model)
        
        stiffness_current = stiffness_current.numpy()[idx]
        stiffness_grad = stiffness_grad.numpy()[idx]
        dOdE = dOdE.numpy()[idx]

        obj = (np.dot(stiffness_current - stiffness_targets_sub, np.transpose(stiffness_current - stiffness_targets_sub)) * 0.5)
        
        grad = np.zeros((n_tiling_params))
        
        for i in range(len(idx)):
            grad += (stiffness_current[i] - stiffness_targets_sub[i]) * stiffness_grad[i].flatten() + \
                (stiffness_current[i] - stiffness_targets_sub[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten()
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    if (not plot_GT) and (not plot_sim):
        tic = time.perf_counter()
        result = minimize(objAndGradient, ti, method='trust-constr', jac=True, options={'disp' : True}, bounds=bounds)
        # result = minimize(objAndGradient, ti, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
        toc = time.perf_counter()
        print(f"Optimization takes {toc - tic:0.6f} seconds")
        uniaxial_strain_opt = []
        for theta in thetas:
            uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, result.x, False)
            uniaxial_strain_opt.append(uni_strain)

        uniaxial_strain_opt = np.reshape(uniaxial_strain_opt, (batch_dim, 3))
        nn_inputs = tf.convert_to_tensor(np.hstack((np.tile(result.x, (batch_dim, 1)), uniaxial_strain_opt)))
        stiffness_opt = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                        tf.convert_to_tensor(thetas), model)
        stiffness_opt = stiffness_opt.numpy()
        print(result.x)
        
        f = open(base_folder + "stiffness_log_IH"+str(IH)+".txt", "w+")
        for i in range(n_tiling_params - 1):
            f.write(str(result.x[i]) + " ")
        f.write(str(result.x[-1]) + "\n")
        f.write(str(len(uniaxial_strain_opt)) + "\n")
        for i in range(len(uniaxial_strain_opt)):
            f.write(str(uniaxial_strain_opt[i][0]) + " " + str(uniaxial_strain_opt[i][1]) + " " + str(uniaxial_strain_opt[i][2]) + "\n")
        f.close()
        # f.write(str(strain + 1.0) + "\n")
        f.close()
    if (not plot_GT) and plot_sim:
        f = open(base_folder + "stiffness_log_IH"+str(IH)+".txt")
        param_opt = [np.float64(i) for i in f.readline().split(" ")]
        uniaxial_strain_opt = []
        for theta in thetas:
            uni_strain, _ = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, param_opt, False)
            uniaxial_strain_opt.append(uni_strain)

        uniaxial_strain_opt = np.reshape(uniaxial_strain_opt, (batch_dim, 3))
        nn_inputs = tf.convert_to_tensor(np.hstack((np.tile(param_opt, (batch_dim, 1)), uniaxial_strain_opt)))
        stiffness_opt = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                        tf.convert_to_tensor(thetas), model)
        stiffness_opt = stiffness_opt.numpy()
        f.close()
        f = open(base_folder + "IH_"+str(IH)+"_stiffness_sim.txt")
        stiffness_sim = [np.float64(i) for i in f.readline().split(" ")]


    def fdGradient(x0):
        eps = 5e-4
        _, grad = objAndGradient(x0)
        print(grad)
        E0, _ = objAndGradient(np.array([x0[0] - eps, x0[1]]))
        E1, _ = objAndGradient(np.array([x0[0] + eps, x0[1]]))
        fd_grad = []
        fd_grad.append((E1 - E0)/2.0/eps)
        E0, _ = objAndGradient(np.array([x0[0], x0[1] - eps]))
        E1, _ = objAndGradient(np.array([x0[0], x0[1] + eps]))
        fd_grad.append((E1 - E0)/2.0/eps)
        print(grad)
        print(fd_grad)

    # fdGradient(ti)
    # exit(0)

    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] + np.pi)
        stiffness = np.append(stiffness, stiffness[i])
        stiffness_targets = np.append(stiffness_targets, stiffness_targets[i])
        if not plot_GT:
            stiffness_opt = np.append(stiffness_opt, stiffness_opt[i])
        if plot_sim:
            stiffness_sim = np.append(stiffness_sim, stiffness_sim[i])
    thetas = np.append(thetas, thetas[0])
    if plot_sim:
        stiffness_sim = np.append(stiffness_sim, stiffness_sim[0])
    stiffness = np.append(stiffness, stiffness[0])
    stiffness_targets = np.append(stiffness_targets, stiffness_targets[0])


    if not plot_GT:
        stiffness_opt = np.append(stiffness_opt, stiffness_opt[0])
    
    min_target, max_target = np.min(stiffness_targets), np.max(stiffness_targets)
    min_init, max_init = np.min(stiffness), np.max(stiffness)
    if not plot_GT:
        min_opt, max_opt = np.min(stiffness_opt), np.max(stiffness_opt)
        max_stiffness = np.max([max_init, max_opt, max_target])
        min_stiffness = np.min([min_init, min_opt, min_target])
    else:
        max_stiffness = np.max([max_init, max_target])
        min_stiffness = np.min([min_init, min_target])
    
    dpr = max_stiffness - min_stiffness

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
    ax1.set_ylim(min_stiffness - 0.1 * dpr, max_stiffness + 0.1 * max_stiffness)
    # if IH == 21:
    #     ax1.set_ylim(0.05, 0.35)
    # elif IH == 1:
    #     ax1.set_ylim(0, 5.5)
    ax1.plot(thetas,stiffness,lw=2.5, label = "stiffness initial", zorder = 0,  color= "#00ABBD")
    ax1.plot(thetas,stiffness_targets,lw=2.5, label = "stiffness target", linestyle = "dashed", color= "#FF9933", zorder = 2)
    # plt.polar(thetas, stiffness, label = "stiffness initial", linewidth=3.0, zorder=0)
    # plt.polar(thetas, stiffness_targets, linestyle = "dashed", label = "stiffness target", linewidth=3.0, zorder=0)
    plt.legend(loc='upper left')


    base_dir = "../results/stiffness/"
    plt.savefig(base_dir+"stiffness_optimization_IH"+str(IH)+"_initial.png", dpi=300)
    plt.close()
    os.system("convert "+base_dir+"stiffness_optimization_IH"+str(IH)+"_initial.png -trim "+base_dir+"stiffness_optimization_IH"+str(IH)+"_initial.png")
    if not plot_GT:
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
        ax1.set_ylim(min_stiffness - 0.1 * dpr, max_stiffness + 0.1 * max_stiffness)
        # if IH == 21:
        #     ax1.set_ylim(0.05, 0.35)
        # elif IH == 1:
        #     ax1.set_ylim(0, 5.5)
        
        
        ax1.plot(thetas,stiffness_opt,lw=2.5, label = "stiffness optimized", zorder = 0,  color= "#00ABBD")
        ax1.plot(thetas,stiffness_targets,lw=2.5, label = "stiffness target", linestyle = "dashed", color= "#FF9933", zorder = 2)
        if plot_sim:
            ax1.plot(thetas,stiffness_sim,lw=2.5, label = "stiffness simulation", linestyle = "dotted", color= "#0099DD", zorder = 3)
        plt.legend(loc='upper left')
        plt.savefig(base_dir+"stiffness_optimization_IH"+str(IH)+"_optimized.png", dpi=300)
        plt.close()
        os.system("convert "+base_dir+"stiffness_optimization_IH"+str(IH)+"_optimized.png -trim "+base_dir+"stiffness_optimization_IH"+str(IH)+"_optimized.png")

def generateStiffnessPlotsForAnimation(IH):
    
    bounds = []
    n_sp_theta = 50
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    strain = 0.02
    current_dir = os.path.dirname(os.path.realpath(__file__))
    idx = np.arange(0, len(thetas), 5)

    if IH == 21:
        strain = 0.02
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti = np.array([0.18, 0.7])
        ti_target = np.array([0.1045, 0.65])
        sample_idx = [2, 7, -1]
        theta = 0.0

    elif IH == 50:
        strain = 0.05
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti = np.array([0.2308, 0.5])
        ti_target = np.array([0.2903, 0.6714])
        
    elif IH == 67:
        strain = 0.1
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        ti = np.array([0.18, 0.68])
        ti_target = np.array([0.25, 0.85])
    elif IH == 22:
        strain = 0.02
        n_tiling_params = 3
        bounds.append([0.1, 0.3])
        bounds.append([0.3, 0.7]) 
        bounds.append([0.0, 0.3])
        ti_target = np.array([0.14, 0.6, 0.3])
        ti = np.array([0.12, 0.5, 0.22])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 29:
        strain = 0.2
        n_tiling_params = 1
        bounds.append([0.005, 1.0])
        ti_target = np.array([0.09])
        ti = np.array([0.2])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 28:
        strain = 0.02
        strain = CauchyToGreen(strain)
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        ti_target = np.array([0.4, 0.6])
        ti = np.array([0.2, 0.6])
        # ti_target = np.array([0.6, 0.6])
        idx = np.arange(0, len(thetas), 5)
    elif IH == 1:
        strain = 0.05
        # strain = CauchyToGreen(strain)
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        # test 1
        ti = np.array([0.1224, 0.5, 0.1434, 0.625])
        # ti_target = np.array([0.1224, 0.6, 0.13, 0.625])
        # test 2
        # ti = np.array([0.1224, 0.6, 0.13, 0.625])
        # ti_target = np.array([0.13, 0.4998, 0.11, 0.6114])
        # test 3
        # ti = np.array([0.13, 0.4998, 0.11, 0.6114])
        # ti_target = np.array([0.1224, 0.5, 0.1087, 0.5541])
        # test 4
        # ti = np.array([0.1224, 0.5, 0.1087, 0.55408])
        # ti_target = np.array([0.1224, 0.5, 0.1434, 0.625])
        # test 5
        # ti = np.array([0.1692, 0.4223, 0.0635, 0.6888])
        # ti_target = np.array([0.1224, 0.4724, 0.12, 0.625])
        # test 6
        # ti = np.array([0.1224, 0.4724, 0.12, 0.625])
        # ti_target = np.array([0.16, 0.5, 0.12, 0.55])
        # test 7
        # ti = np.array([0.16, 0.5, 0.12, 0.55])
        # ti_target = np.array([0.22, 0.6, 0.08, 0.6])
        # test 8
        # ti = np.array([0.18710856, 0.58457689, 0.10264114, 0.74953785])
        # ti_target = np.array([0.1692, 0.4223, 0.0635, 0.6888])
        # 0.19396568 0.46893408 0.06722148 0.75063715

        # ti = np.array([0.2434, 0.4494, 0.0494, 0.625])
        # ti = np.array([0.16, 0.5, 0.12, 0.55])
        ti_target = np.array([0.26, 0.75,  0.15,  0.58])
        # ti_target = np.array([0.1949, 0.6434, 0.1403, 0.6858])
        idx = np.arange(0, len(thetas), 5)

    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)

    model_name += "double"
    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    if use_double:
        model = buildNMNModel(n_tiling_params, tf.float64)
    else:
        model = buildNMNModel(n_tiling_params, tf.float32)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    base_dir = "../results/stiffness/"
    stiffness_init = generateStiffnessDataThetas(thetas, n_tiling_params, strain, ti, model)
    
    f = open(base_dir + "stiffness_log_IH"+str(IH)+".txt")
        
    param_opt = [np.float64(i) for i in f.readline().split(" ")]

    f.close()

    stiffness_opt = generateStiffnessDataThetas(thetas, n_tiling_params, strain, param_opt, model)


    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] + np.pi)
        stiffness_init = np.append(stiffness_init, stiffness_init[i])
        stiffness_opt = np.append(stiffness_opt, stiffness_opt[i])
        
    thetas = np.append(thetas, thetas[0])
    stiffness_init = np.append(stiffness_init, stiffness_init[0])
    stiffness_opt = np.append(stiffness_opt, stiffness_opt[0])

    
    min_target, max_target = np.min(stiffness_opt), np.max(stiffness_opt)
    min_init, max_init = np.min(stiffness_init), np.max(stiffness_init)
    
    min_opt, max_opt = np.min(stiffness_opt), np.max(stiffness_opt)
    max_stiffness = np.max([max_init, max_opt, max_target])
    min_stiffness = np.min([min_init, min_opt, min_target])
    
    
    dpr = max_stiffness - min_stiffness

    for i in range(len(stiffness_init)):
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
        ax1.set_ylim(min_stiffness - 0.1 * dpr, max_stiffness + 0.1 * max_stiffness)
        
        ax1.plot(thetas,stiffness_init,lw=2.5, label = "stiffness initial", color=(65/255.0, 143/255.0, 240/255.0))
        ax1.plot(thetas,stiffness_opt,lw=2.5, label = "stiffness optimized", color=(245/255.0, 189/255.0, 65/255.0))

        ax1.scatter(thetas[i],stiffness_init[i],s=20, zorder = 5, color=(65/255.0, 143/255.0, 240/255.0))
        ax1.scatter(thetas[i],stiffness_opt[i],s=20, zorder = 5, color=(245/255.0, 189/255.0, 65/255.0))
        plt.legend(loc="upper left")
        plt.savefig(base_dir+"/IH"+str(IH)+"/stiffness_optimization_IH"+str(IH)+"_animation"+str(i)+".png", dpi=300)
        # plt.savefig(base_dir+"/test"+"/stiffness_optimization_IH"+str(IH)+"_animation"+str(i)+".png", dpi=300)
        
        plt.close()
        # exit(0)
        # os.system("convert "+base_dir+"stiffness_optimization_IH"+str(IH)+"_optimized.png -trim "+base_dir+"stiffness_optimization_IH"+str(IH)+"_optimized.png")


def getDirectionStiffness(ti, n_tiling_params, model, strain_cauchy, n_sp_theta = 20, sym=True):
    if strain_cauchy <  0:
        strain = strain_cauchy - 0.5 * strain_cauchy  * strain_cauchy
    else:
        strain = strain_cauchy + 0.5 * strain_cauchy  * strain_cauchy
    
    thetas = np.arange(0.0, np.pi, np.pi/float(n_sp_theta))
    uniaxial_strain = computeUniaxialStrainThetaBatch(n_tiling_params, strain, thetas, model, ti, True)
    
    batch_dim = len(thetas)
    ti_batch = np.tile(ti, (batch_dim, 1))
    nn_inputs = tf.convert_to_tensor(np.hstack((ti_batch, uniaxial_strain)))
    stiffness = computeDirectionalStiffness(n_tiling_params, nn_inputs, 
                    tf.convert_to_tensor(thetas), model)
    stiffness = stiffness.numpy()
    if sym:
        for i in range(n_sp_theta):
            thetas= np.append(thetas, thetas[i] + np.pi)
            stiffness = np.append(stiffness, stiffness[i])
        thetas= np.append(thetas, 2*np.pi)
        stiffness = np.append(stiffness, stiffness[0])
    return thetas, stiffness

def fillPolarData(thetas, stiffness):
    n_sp_theta = len(thetas)
    for i in range(n_sp_theta):
        thetas= np.append(thetas, thetas[i] + np.pi)
        stiffness = np.append(stiffness, stiffness[i])
    # thetas= np.append(thetas, thetas[0] + 2*np.pi)
    # stiffness = np.append(stiffness, stiffness[0])
    return thetas, stiffness

def stiffnessModifyUI():
    IH = 28
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    ti = np.array([0.33771952, 0.48740965])
    
    thetas_nn = np.arange(0.0, np.pi, np.pi/float(50))
    thetas = np.arange(0.0, np.pi, np.pi/float(50))
    
    thetas, stiffness = getDirectionStiffness(ti, n_tiling_params, model, 0.05, 20, False)

    # stiffness=generateStiffnessDataThetas(thetas, n_tiling_params, 0.1, ti, model)

    thetas_full, stiffness_full = fillPolarData(thetas, stiffness)

    # x, y = pol2cart(stiffness_full, thetas_full)
    x, y = thetas_full, stiffness_full

    min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)
    dx, dy = max_x - min_x, max_y - min_y

    poly = Polygon(np.column_stack([x, y]), animated=True, visible = False)


    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    fig.set_size_inches(20, 20)
    ax.add_patch(poly)
    p = MacroPropertyModifier(ax, poly, thetas_full, thetas_nn)
    # ax.set_title('Move control points in Cartesian space')
    # ax.set_xlim((min_x - 0.2 * dx, max_x + 0.2 * dx))
    # ax.set_ylim((min_y - 0.2 * dy, max_y + 0.2 * dy))    
    # ax.set_ylim((min_y - 0.05 * dy, max_y + 0.05 * dy))    
    ax.set_ylim(0, 1.4)
    ax.grid(linewidth=3)

    # plt.axis('off')
    # plt.polar([], [])
    plt.show()

if __name__ == "__main__":
    
    for idx in [1]:
        stiffnessOptimizationSA(idx, False, False)
    # for idx in [21, 22, 28, 29, 50, 67]:
        # generateStiffnessPlotsForAnimation(idx)
    # stiffnessModifyUI()