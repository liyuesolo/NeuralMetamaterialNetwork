

import os
from functools import cmp_to_key
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import numpy as np
import tensorflow as tf
from model import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import time
from Derivatives import *
from Optimization import *
from Common import *

use_double = True
if use_double:
    tf.keras.backend.set_floatx("float64")
else:
    tf.keras.backend.set_floatx("float32")

######################## NeoHookean Test Start ########################
# in case you would like to check against analytical data
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
    C = tape_outer.batch_jacobian(stress, strain)
    return C, stress, psi


def optimizeUniaxialStrainNHSingleDirection(theta, strain):
    data_type = tf.float32
    d = np.array([np.cos(theta), np.sin(theta)])
    strain_tensor_init = np.outer(d, d) * strain
    strain_init = np.array([strain_tensor_init[0][0], strain_tensor_init[1][1], 2.0 * strain_tensor_init[0][1]])
    n = np.array([-np.sin(theta), np.cos(theta)])
    def constraint(x):
        strain_tensor = np.reshape([x[0], 0.5 * x[-1], 0.5 * x[-1], x[1]], (2, 2))
        dTEd = np.dot(d, np.dot(strain_tensor, np.transpose(d)))
        c = dTEd - strain
        # print("c", c)
        return c

    def hessian(x):
        C, stress, psi = psiGradHessNH(tf.convert_to_tensor([x], dtype=data_type), data_type)
        H = C[0].numpy()
        alpha = 1e-6
        while not np.all(np.linalg.eigvals(H) > 0):
            H += np.diag(np.full(3,alpha))
            alpha *= 10.0
        # print(H)
        return H

    def objAndEnergy(x):
        C, stress, psi = psiGradHessNH(tf.convert_to_tensor([x], dtype=data_type), data_type)
        
        obj = np.squeeze(psi.numpy()) 
        grad = stress.numpy().flatten()
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
    result = minimize(objAndEnergy, strain_init, method='trust-constr', jac=True, hess=hessian,
        constraints={"fun": constraint, "type": "eq"},
        options={'disp' : True})
    
    return result.x

######################## NeoHookean Test End ########################

######################## Main Function Start ########################
@tf.function
def objUniaxialStress(n_tiling_params, ti, uniaxial_strain, theta, model):
    batch_dim = uniaxial_strain.shape[0]
    thetas = tf.tile(theta, (uniaxial_strain.shape[0], 1))
    
    d = tf.concat((tf.math.cos(thetas),
                        tf.math.sin(thetas)), 
                        axis = 1)
    d = tf.cast(d, tf.float64)
    ti = tf.expand_dims(ti, 0)
    
    with tf.GradientTape() as tape:
        
        ti_batch = tf.tile(ti, (batch_dim, 1))
        inputs = tf.concat((ti_batch, uniaxial_strain), axis=1)
        tape.watch(inputs)
        psi = model(inputs, training=False)
        dedlambda = tape.gradient(psi, inputs)
        stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        stress_xx = tf.gather(stress, [0], axis = 1)
        stress_yy = tf.gather(stress, [1], axis = 1)
        stress_xy = tf.gather(stress, [2], axis = 1)
        stress_reorder = tf.concat((stress_xx, stress_xy, stress_xy, stress_yy), axis=1)
        stress_tensor = tf.reshape(stress_reorder, (batch_dim, 2, 2))
        Sd = tf.linalg.matvec(stress_tensor, d)
        
        dTSd = tf.einsum("ij,ij->i",d, Sd)
    del tape
    return tf.squeeze(dTSd)


def optimizeUniaxialStressSA(IH, plot_sim = False, plot_GT = False, save_data=False):
    bounds = []
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    theta = 0.0
    strain_range = [-0.1, 0.2]
    n_sp_strain = 25

    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))

    # strain_samples = np.arange(strain_range[0], strain_range[1], 0.01)
    # for i in range(len(strain_samples)):
        # strain_samples[i] = CauchyToGreen(strain_samples[i])
        # if strain < 0:
        #     strain_samples[i] = strain - 0.5 * strain * strain
        # else:
        #     strain_samples[i] = strain + 0.5 * strain * strain

    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        # ti0 = np.array([0.16, 0.52])
        ti_target = np.array([0.106, 0.65])
        ti0 = np.array([0.115, 0.765])
        # sample_idx = np.arange(2, n_sp_strain-2, 6)
        theta = 0.0
    elif IH == 50:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti0 = np.array([0.2, 0.52])
        # ti0 = np.array([0.2903, 0.6714])
        ti_target = np.array([0.21, 0.6])
        theta = 0.0 * np.pi
    elif IH == 67:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        ti0 = np.array([0.24, 0.87])
        ti_target = np.array([0.15, 0.74])
        # sample_idx = [2, 7, -1]
        theta = 0.5 * np.pi
    elif IH == 28:
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        # ti0 = np.array([0.4528, 0.5])
        # ti0 = np.array([0.4, 0.8])
        
        # ti0 = np.array([0.3, 0.5])
        ti0 = np.array([0.03411184, 0.37176683])
        ti_target = np.array([0.2205, 0.6016])
        sample_idx = np.arange(0, n_sp_strain, 6)
        theta = 0.5 * np.pi
    elif IH == 1:
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        ti_target = np.array([0.1224, 0.5, 0.1434, 0.625])
        # ti0 = np.array([0.1, 0.5, 0.13, 0.45])
        ti0 = np.array([0.1224, 0.6, 0.1434, 0.625])
        # ti0 = np.array([ 0.12, 0.504, 0.1, 0.625])
        theta = 0.25 * np.pi
    elif IH == 29:
        n_tiling_params = 1
        bounds.append([0.005, 1.0])
        ti_target = np.array([0.45])
        ti0 = np.array([0.15])
        theta = 1.0 / 4.0 * np.pi
    elif IH == 22:
        n_tiling_params = 3
        bounds.append([0.1, 0.3])
        bounds.append([0.3, 0.7]) 
        bounds.append([0.0, 0.3])
        ti_target = np.array([0.2, 0.6, 0.12])
        ti0 = np.array([0.2, 0.7, 0.15])
        theta = 0.0 * np.pi

    sample_idx = np.arange(5, n_sp_strain-3, 8)
    # theta = 0.5 * np.pi
    
    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)
    
    if use_double:
        model_name += "double"

    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    model = buildNMNModel(n_tiling_params)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    
    base_dir = "../results/strain_stress/"
    
    if (plot_sim):
        sim_file = base_dir + "/IH_" + str(IH) + "_strain_stress_sim.txt"
        sim_obj = loadSimulationDataSorted(sim_file, theta)
    

    def computeTarget(ti):
        
        uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                    theta, strain_samples, 
                                    ti)
        
        
        ti_TF = tf.convert_to_tensor(ti)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        obj = objUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        obj = obj.numpy()
        return obj

    obj_init = computeTarget(ti0)
    obj_target = computeTarget(ti_target)

    # if plotting the target curve
    if plot_GT:
        info = ""
        for data in obj_init[sample_idx]:
            info += str(data) + ", "
        print(info)
        info = ""
        for data in strain_samples[sample_idx]:
            info += str(data) + ", "
        print(info)
        # exit(0)
    
    
    stress_targets = obj_target[sample_idx]
    
    def objAndGradient(x):
        uniaxial_strain = []
        dqdp = []
        # tic = time.perf_counter()
        for strain in strain_samples[sample_idx]:
            uni_strain, dqidpi = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, x, False)
            uniaxial_strain.append(uni_strain)
            dqdp.append(dqidpi)
        # toc = time.perf_counter()
        # print(f"constraint takes {toc - tic:0.6f} seconds")
        ti_TF = tf.convert_to_tensor(x)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        stress_d, stress_d_grad, dOdE = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        
        stress_d = stress_d.numpy()
        stress_d_grad = stress_d_grad.numpy()
        dOdE = dOdE.numpy()
        
        stress_current = stress_d
        obj = (np.dot(stress_current - stress_targets, np.transpose(stress_current - stress_targets)) * 0.5).flatten() #/ np.linalg.norm(stress_targets)
        grad = np.zeros((n_tiling_params))
        
        for i in range(len(sample_idx)):
            grad += (stress_current[i] - stress_targets[i]) * stress_d_grad[i].flatten() + \
                (stress_current[i] - stress_targets[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten() #/ np.linalg.norm(stress_targets)
        
        # print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad
    
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
    
    
    if (not plot_GT) and (not plot_sim):
        result = minimize(objAndGradient, ti0, method='trust-constr', jac=True, options={'disp' : True}, bounds=bounds)
        tic = time.perf_counter()
        # result = minimize(objAndGradient, ti0, method='L-BFGS-B', jac=True, options={'disp' : False, "iprint": -1}, bounds=bounds)
        toc = time.perf_counter()
        print(f"Optimization takes {toc - tic:0.6f} seconds")
        
        uniaxial_strain_opt = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                        theta, strain_samples, 
                        result.x, model)
        obj_opt = objUniaxialStress(n_tiling_params, tf.convert_to_tensor(result.x), tf.convert_to_tensor(uniaxial_strain_opt), tf.constant([[theta]]), model)
        obj_opt = obj_opt.numpy()
        print("final obj: "result.fun)
        print("tiling params: ",result.x)
        if save_data:
            f = open(base_dir + "uniaxial_stress_IH"+str(IH)+".txt", "w+")
            for i in range(n_tiling_params - 1):
                f.write(str(result.x[i]) + " ")
            f.write(str(result.x[-1]) + "\n")
            f.write(str(len(strain_samples)) + "\n")
            for i in range(len(uniaxial_strain_opt)):
                f.write(str(uniaxial_strain_opt[i][0]) + " " + str(uniaxial_strain_opt[i][1]) + " " + str(uniaxial_strain_opt[i][2]) + "\n")
            f.close()

    if (not plot_GT) and plot_sim:
        f = open(base_dir + "uniaxial_stress_IH"+str(IH)+".txt")
        
        param_opt = [np.float64(i) for i in f.readline().split(" ")]
        uniaxial_strain_opt = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                        theta, strain_samples, 
                        param_opt, model)
        obj_opt = objUniaxialStress(n_tiling_params, tf.convert_to_tensor(param_opt), tf.convert_to_tensor(uniaxial_strain_opt), tf.constant([[theta]]), model)
        obj_opt = obj_opt.numpy()
        f.close()
    if plot_GT:
        if save_data:
            f = open(base_dir + "uniaxial_stress_IH"+str(IH)+".txt", "w+")
            for i in range(n_tiling_params - 1):
                f.write(str(ti0[i]) + " ")
            f.write(str(ti0[-1]) + "\n")
            uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                        theta, strain_samples, 
                                        ti0)
            f.write(str(len(strain_samples)) + "\n")
            for i in range(len(uniaxial_strain)):
                f.write(str(uniaxial_strain[i][0]) + " " + str(uniaxial_strain[i][1]) + " " + str(uniaxial_strain[i][2]) + "\n")
            f.close()

    for i in range(len(strain_samples)):
        strain_samples[i] = strain_samples[i] * 100.0

    strain_points = strain_samples[sample_idx]
    if save_data:
        plt.plot(strain_samples, obj_init, label="stress initial", linewidth=3.0, zorder=0, color= "#00ABBD")
        if not plot_GT:
            plt.plot(strain_samples, obj_opt, label = "stress optimized", linewidth=3.0, zorder=0, color= "#A1C7E0")
        if plot_sim:
            plt.plot(strain_samples, sim_obj, label = "stress simulation", linewidth=3.0, linestyle='dotted', zorder=2, color= "#0099DD")
        plt.scatter(strain_points, stress_targets, marker='+', s=200.0, label = "targets", c="#FF9933", zorder=5)
        plt.legend(loc="upper left")
    # plt.xlabel("strain")
    # plt.ylabel("stress")
        plt.savefig(base_dir+"uniaxial_stress_IH"+str(IH)+".png", dpi=300)
        plt.close()

######################## Main Function End ########################


######################## Somewhat Useful Scripts Start ########################
def plotNNFDCurves():
    bounds = []
    IH = 21
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    theta = 0.0

    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        ti_init = np.array([0.115, 0.765])
        ti_NN = np.array([0.11611517, 0.6579828])
        ti_LBFGS = np.array([0.105, 0.79499])
        ti_MMA = np.array([0.101928, 0.509179])
        ti_GD = np.array([0.1, 0.54811])
        sample_idx = [2, 7, -1]
        theta = 0.0
    
    
    save_path = os.path.join(current_dir, 'Models/IH' + str(IH) + "/")
    model = buildNMNModel(n_tiling_params)
    model.load_weights(save_path + "IH"+str(IH) + '.tf')

    
    strain_range = [-0.05, 0.1]
    strain_samples = np.arange(strain_range[0], strain_range[1], 0.01)
    for i in range(len(strain_samples)):
        strain = strain_samples[i]
        if strain < 0:
            strain_samples[i] = strain - 0.5 * strain * strain
        else:
            strain_samples[i] = strain + 0.5 * strain * strain
    
    def obj(ti):
        uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                    theta, strain_samples, 
                                    ti, model)
        ti_TF = tf.convert_to_tensor(ti)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        obj_init, _ , _ = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        obj_init = obj_init.numpy()
        return obj_init
    
    obj_NN_LBFGS = obj(ti_NN)
    obj_FD_LBFGS = obj(ti_LBFGS)
    obj_FD_MMA = obj(ti_MMA)
    obj_FD_GD = obj(ti_GD)
    obj_init = obj(ti_init)


    for i in range(len(strain_samples)):
        strain_samples[i] = strain_samples[i] * 100.0

    strain_points = strain_samples[sample_idx]
    stress_targets = [-0.00598749,  0.00477436,  0.04006726]
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(strain_samples, obj_init, label="initial guess", linewidth=2.0, zorder=0)
    plt.plot(strain_samples, obj_FD_GD, label="FD-PGD", linewidth=2.0, zorder=0, color = "#00ABBD")
    plt.plot(strain_samples, obj_FD_MMA, label="FD-MMA", linewidth=2.0, zorder=0, color = "#48D1CC")
    plt.plot(strain_samples, obj_FD_LBFGS, label="FD-LBFGS-B", linewidth=2.0, zorder=0, color = "#026E81")
    plt.plot(strain_samples, obj_NN_LBFGS, label="NN-LBFGS-B", linewidth=2.0, zorder=0, color = "#FF9933")
    plt.scatter(strain_points, stress_targets, marker='+', s=200.0, zorder=5, color = "red", label = "targets")
    plt.legend(loc="upper left")
    plt.savefig("NN_FD_comparison.png", dpi=300)
    os.system("convert NN_FD_comparison.png -trim NN_FD_comparison.png")
    plt.close()


def loadSimulationDataSorted(filename, theta):
    
    
    sim_obj = []
    strain_mag = []
    for line in open(filename).readlines():
        item = [np.float64(i) for i in line.strip().split(" ")]
        
        d = np.array([np.cos(theta), np.sin(theta)])
        stress_voigt = item[-4:-1]
        stress = np.array([[stress_voigt[0], stress_voigt[2]], [stress_voigt[2], stress_voigt[1]]])
        sim_obj.append(d.dot(stress.dot(d)))
        strain_mag.append(item[-3])
    return sim_obj    


def plotCurveForAnimation(IH):
    bounds = []
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    theta = 0.0
    
    strain_range = [-0.1, 0.2]
    n_sp_strain = 25
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    # strain_samples = np.arange(strain_range[0], strain_range[1], 0.01)
    # for i in range(len(strain_samples)):
        # strain_samples[i] = CauchyToGreen(strain_samples[i])
        # if strain < 0:
        #     strain_samples[i] = strain - 0.5 * strain * strain
        # else:
        #     strain_samples[i] = strain + 0.5 * strain * strain

    if IH == 21:
        n_tiling_params = 2
        bounds.append([0.105, 0.195])
        bounds.append([0.505, 0.795])
        # ti0 = np.array([0.16, 0.52])
        ti_target = np.array([0.106, 0.65])
        ti0 = np.array([0.115, 0.765])
        # sample_idx = np.arange(2, n_sp_strain-2, 6)
        theta = 0.0
    elif IH == 50:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.25, 0.75])
        ti0 = np.array([0.2, 0.52])
        # ti0 = np.array([0.2903, 0.6714])
        ti_target = np.array([0.21, 0.6])
        theta = 0.0 * np.pi
    elif IH == 67:
        n_tiling_params = 2
        bounds.append([0.1, 0.3])
        bounds.append([0.6, 1.1])
        ti0 = np.array([0.24, 0.87])
        ti_target = np.array([0.15, 0.74])
        # sample_idx = [2, 7, -1]
        theta = 0.5 * np.pi
    elif IH == 28:
        n_tiling_params = 2
        bounds.append([0.005, 0.8])
        bounds.append([0.005, 1.0])
        # ti0 = np.array([0.4528, 0.5])
        # ti0 = np.array([0.4, 0.8])
        
        # ti0 = np.array([0.3, 0.5])
        ti0 = np.array([0.03411184, 0.37176683])
        ti_target = np.array([0.2205, 0.6016])
        sample_idx = np.arange(0, n_sp_strain, 6)
        theta = 0.5 * np.pi
    elif IH == 1:
        n_tiling_params = 4
        bounds.append([0.05, 0.3])
        bounds.append([0.25, 0.75])
        bounds.append([0.05, 0.15])
        bounds.append([0.4, 0.8])
        ti_target = np.array([0.1224, 0.5, 0.1434, 0.625])
        # ti0 = np.array([0.1, 0.5, 0.13, 0.45])
        ti0 = np.array([0.1224, 0.6, 0.1434, 0.625])
        # ti0 = np.array([ 0.12, 0.504, 0.1, 0.625])
        theta = 0.25 * np.pi
    elif IH == 29:
        n_tiling_params = 1
        bounds.append([0.005, 1.0])
        ti_target = np.array([0.45])
        ti0 = np.array([0.15])
        theta = 1.0 / 4.0 * np.pi
    elif IH == 22:
        n_tiling_params = 3
        bounds.append([0.1, 0.3])
        bounds.append([0.3, 0.7]) 
        bounds.append([0.0, 0.3])
        ti_target = np.array([0.2, 0.6, 0.12])
        ti0 = np.array([0.2, 0.7, 0.15])
        theta = 0.0 * np.pi

    sample_idx = np.arange(5, n_sp_strain-3, 8)
    # theta = 0.5 * np.pi
    
    model_name = str(IH)
    if IH < 10:
        model_name = "0" + str(IH)
    else:
        model_name = str(IH)
    
    if use_double:
        model_name += "double"

    save_path = os.path.join(current_dir, 'Models/IH' + model_name + "/")
    model = buildNMNModel(n_tiling_params)
    model.load_weights(save_path + "IH" + model_name + '.tf')

    
    base_dir = "../results/strain_stress/"

    def computeTarget(ti):
        
        uniaxial_strain = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                                    theta, strain_samples, 
                                    ti)
        
        # for i in range(len(uniaxial_strain)):
        #     print(uniaxial_strain[i], 1.0 + strain_samples[i], theta, ti)
        ti_TF = tf.convert_to_tensor(ti)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        obj = objUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        obj = obj.numpy()
        return obj

    obj_init = computeTarget(ti0)

    f = open(base_dir + "uniaxial_stress_IH"+str(IH)+".txt")
        
    param_opt = [np.float64(i) for i in f.readline().split(" ")]
    
    uniaxial_strain_opt = optimizeUniaxialStrainSingleDirectionConstraintBatch(model, n_tiling_params, 
                    theta, strain_samples, 
                    param_opt, model)
    obj_opt = objUniaxialStress(n_tiling_params, tf.convert_to_tensor(param_opt), tf.convert_to_tensor(uniaxial_strain_opt), tf.constant([[theta]]), model)
    obj_opt = obj_opt.numpy()
    f.close()

    x_min = np.min(strain_samples)
    x_max = np.max(strain_samples)
    dx = x_max - x_min

    y_min = np.minimum(np.min(obj_init), np.min(obj_opt))
    y_max = np.maximum(np.max(obj_init), np.max(obj_opt))
    dy = y_max - y_min

    for i in range(len(strain_samples)):
        plt.plot(strain_samples, obj_init, label="structure initial", linewidth=3.0, zorder=0, color=(65/255.0, 143/255.0, 240/255.0))
        plt.plot(strain_samples, obj_opt, label="structure optimized", linewidth=3.0, zorder=0, color=(245/255.0, 189/255.0, 65/255.0))
        plt.ylim([y_min - 0.1 * dy, y_max + 0.1 * dy])
        plt.xlim([x_min - 0.1 * dx , x_max + 0.1 * dx])
        plt.scatter([strain_samples[i]], [obj_init[i]], zorder=5, color=(65/255.0, 143/255.0, 240/255.0))
        plt.scatter([strain_samples[i]], [obj_opt[i]], zorder=5, color=(245/255.0, 189/255.0, 65/255.0))
        plt.legend(loc="upper left")
        plt.savefig(base_dir+"/IH"+str(IH)+"/uniaxial_stress_IH"+str(IH)+"_animation_"+str(i)+".png", dpi=300)
        plt.close()



def optimizeFromInitialGuess(model, ti, bounds, n_tiling_params, stress_targets):
    strain_range = [-0.1, 0.0]
    n_sp_strain = 30
    strain_samples = np.arange(strain_range[0], strain_range[1], (strain_range[1] - strain_range[0])/float(n_sp_strain))
    sample_idx = np.arange(0, n_sp_strain, 4)
    theta = 0.5 * np.pi

    def objAndGradient(x):
        uniaxial_strain = []
        dqdp = []
        for strain in strain_samples[sample_idx]:
            uni_strain, dqidpi = optimizeUniaxialStrainSingleDirectionConstraint(model, n_tiling_params, theta, strain, x, False)
            uniaxial_strain.append(uni_strain)
            dqdp.append(dqidpi)
        ti_TF = tf.convert_to_tensor(x)
        uniaxial_strain_TF = tf.convert_to_tensor(uniaxial_strain)
        stress_d, stress_d_grad, dOdE = objGradUniaxialStress(n_tiling_params, ti_TF, uniaxial_strain_TF, tf.constant([[theta]]), model)
        
        stress_d = stress_d.numpy()
        stress_d_grad = stress_d_grad.numpy()
        dOdE = dOdE.numpy()
        
        # stress_current = stress_d[sample_idx]#np.array([stress_d[2], stress_d[5], stress_d[-1]])
        stress_current = stress_d
        obj = (np.dot(stress_current - stress_targets, np.transpose(stress_current - stress_targets)) * 0.5).flatten() 
        grad = np.zeros((n_tiling_params))
        
        for i in range(len(sample_idx)):
            grad += (stress_current[i] - stress_targets[i]) * stress_d_grad[i].flatten() + \
                (stress_current[i] - stress_targets[i]) * np.dot(dOdE[i][i], dqdp[i][:3, :]).flatten()
        
        print("obj: {} |grad|: {}".format(obj, np.linalg.norm(grad)))
        return obj, grad

    result = minimize(objAndGradient, ti, method='L-BFGS-B', jac=True, options={'disp' : True}, bounds=bounds)
  
    return result.fun, result.x

def generateSamples(bounds, n_samples):
    n_tiling_params = len(bounds)
    random_samples = np.random.random_sample(n_samples * n_tiling_params)
    samples = []
    for i in range(n_samples):
        sample = np.zeros(n_tiling_params)
        for j in range(n_tiling_params):
            sample[j] = bounds[j][0] + (bounds[j][1] - bounds[j][0]) * random_samples[i * n_tiling_params + j]
        samples.append(sample)  
    return samples


def searchBestFitDifferentInitialGuess(IH):
    model, n_tiling_params, ti_default, bounds = loadModel(IH)
    samples = generateSamples(bounds, 20)
    objectives = []
    # stress_targets = [-0.265, -0.26, -0.25, -0.24, -0.21, -0.16,-0.1, -0.039]
    # stress_targets = np.array([-0.17, -0.162, -0.15, -0.134, -0.11, -0.083, -0.058, -0.03])
    # stress_targets = [-0.309, -0.288, -0.258, -0.235, -0.222, -0.205, -0.148, -0.0532]
    stress_targets = [-0.265, -0.26, -0.255, -0.24, -0.21, -0.16,-0.1, -0.039]
    # stress_targets = np.array([-0.09, -0.082, -0.074, -0.065, -0.055, -0.045, -0.03, -0.005])
    results = []
    for sample in samples:  
        val, sol = optimizeFromInitialGuess(model, sample, bounds, n_tiling_params, stress_targets)
        objectives.append(val)
        results.append(sol)
    
    objectives = np.squeeze(np.array(objectives))

    min_value = 1e10
    min_idx = -1
    for i in range(len(objectives)):
        if objectives[i] < min_value:
            min_value = objectives[i]
            min_idx = i

    print(results[min_idx], objectives[min_idx], samples[min_idx])


def loadSimulationData(filename, n_samples, n_tiling_params):
    samples = []
    thetas = []
    line_cnt = 0
    data_items = []
    for line in open(filename).readlines():
        item = [np.float64(i) for i in line.strip().split(" ")]
        if line_cnt < n_samples:
            samples.append(item[:n_tiling_params])
            thetas.append(item[n_tiling_params])
        else:
            data_items.append(item)
        line_cnt += 1
    
    return samples, thetas, data_items

######################## Somewhat Useful Scripts End ########################


##############################################################################################################
if __name__ == "__main__":

    # these are the tiling family that we trained on
    # for idx in [1, 21, 22, 28, 50, 67]:
    optimizeUniaxialStressSA(21, plot_sim=False, plot_GT=False, save_data=True)
    
    

