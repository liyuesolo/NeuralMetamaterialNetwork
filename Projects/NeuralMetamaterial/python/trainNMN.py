import os
from functools import cmp_to_key

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import math
import numpy as np
import tensorflow as tf
from model import *
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras.backend as K
from Summary import *
import tensorflow_probability as tfp
import scipy

# train with double precision to avoid floating point error
tf.keras.backend.set_floatx('float64')


def relativeL2(y_true, y_pred):
    if (y_true.shape[1] > 1):
        stress_norm = tf.norm(y_true, ord='euclidean', axis=1)
        norm = tf.tile(tf.keras.backend.expand_dims(stress_norm, 1), tf.constant([1, 3]))
        y_true_normalized = tf.divide(y_true, norm)
        y_pred_normalized = tf.divide(y_pred, norm)
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
    else:
        y_true_normalized = tf.divide(y_true, y_true + K.epsilon())
        y_pred_normalized = tf.divide(y_pred, y_true + K.epsilon())
        return K.mean(K.square(y_true_normalized - y_pred_normalized))
        
def loadDataSplitTest(n_tiling_params, filename, shuffle = True, ignore_unconverging_result = True):
    all_data = []
    all_label = [] 
    
    for line in open(filename).readlines():
        item = [float(i) for i in line.strip().split(" ")[:]]
        if (ignore_unconverging_result):
            # check if residual normal is too large
            if (item[-1] > 5e-6 or math.isnan(item[-1])):
                continue
            # avoid dividing by small number
            if (item[-5] < 1e-5 or item[-5] > 10):
                continue
        data = item[0:n_tiling_params] ## tiling parameters
        for i in range(2):
            data.append(item[n_tiling_params+i]) ## strain xx strain yy
        data.append(2.0 * item[n_tiling_params+2]) ## 2.0 * strain xy
        
        # stress xx, yy, xy
        label = item[n_tiling_params+3:n_tiling_params+7]
        
        all_data.append(data)
        all_label.append(label)
        
    print("#valid data:{}".format(len(all_data)))
    
    all_data = np.array(all_data[:]).astype(np.float64)
    all_label = np.array(all_label[:]).astype(np.float64) 
    
    
    indices = np.arange(all_data.shape[0])
    if (shuffle):
        np.random.shuffle(indices)
    
    all_data = all_data[indices]
    all_label = all_label[indices]
    
    return all_data, all_label


def generator(train_data, train_label):    
    indices = np.arange(train_data.shape[0])
    while True:
        np.random.shuffle(indices)
        yield train_data[indices], train_label[indices]

##################### training step #####################
@tf.function
def trainStep(n_tiling_params, opt, lambdas, sigmas, model, train_vars):
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(train_vars)
        tape.watch(lambdas)
        
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 3])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        
        grad_loss = relativeL2(stress_gt, stress_pred)
        e_loss = relativeL2(potential_gt, psi)

        loss = grad_loss + e_loss
        
    dLdw = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(dLdw, train_vars))
    gradNorm = tf.math.sqrt(tf.reduce_sum([tf.reduce_sum(gi*gi) for gi in dLdw]))
    
    del tape
    return grad_loss, e_loss, gradNorm

##################### testing step #####################
@tf.function
def testStep(n_tiling_params, lambdas, sigmas, model):
    
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
        batch_dim = psi.shape[0]
        stress_gt = tf.slice(sigmas, [0, 0], [batch_dim, 3])
        potential_gt = tf.slice(sigmas, [0, sigmas.shape[1]-1], [batch_dim, 1])
        stress_pred = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
        
        grad_loss = relativeL2(stress_gt, stress_pred)
        e_loss = relativeL2(potential_gt, psi)
    del tape
    return grad_loss, e_loss, stress_pred, psi

##################### for timing purpose #####################
@tf.function
def testStepHess(n_tiling_params, lambdas, model):
    with tf.GradientTape() as tape_outer:
        tape_outer.watch(lambdas)
        with tf.GradientTape() as tape:
            tape.watch(lambdas)
            elastic_potential = model(lambdas, training=False)
            dedlambda = tape.gradient(elastic_potential, lambdas)
            batch_dim = elastic_potential.shape[0]
            stress = tf.slice(dedlambda, [0, n_tiling_params], [batch_dim, 3])
            de_dp = tf.slice(dedlambda, [0, 0], [batch_dim, n_tiling_params])
    dstress_dp = tape_outer.batch_jacobian(stress, lambdas)[:, :, 0:n_tiling_params]
    del tape
    del tape_outer


@tf.function
def testStepGrad(n_tiling_params, lambdas, sigmas, model):
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        psi = model(lambdas)
        dedlambda = tape.gradient(psi, lambdas)
    del tape

@tf.function
def testStepValue(n_tiling_params, lambdas, sigmas, model):
    with tf.GradientTape() as tape:
        tape.watch(lambdas)
        psi = model(lambdas)
    del tape

###############################################################

def plot(prefix, prediction, label, gt_only = False):
    def cmp_sigma_xx(i, j):
        return label[i][0] - label[j][0]
    def cmp_sigma_xy(i, j):
        return label[i][2] - label[j][2]
    def cmp_sigma_yx(i, j):
        return label[i][3] - label[j][3]
    def cmp_sigma_yy(i, j):
        return label[i][1] - label[j][1]
        
    indices = [i for i in range(len(label))]
    data_point = [i for i in range(len(label))]

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xx))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xx_gt = [sigma_gt_sorted[i][0] for i in range(len(label))]
    sigma_xx = [sigma_sorted[i][0] for i in range(len(label))]
    if not gt_only:
        plt.plot(data_point, sigma_xx, linewidth=1.0, label = "Sigma_xx")
    plt.plot(data_point, sigma_xx_gt, linewidth=1.0, label = "GT Sigma_xx")
    plt.legend(loc="upper left")
    plt.savefig(prefix+"_learned_sigma_xx.png", dpi = 300)
    plt.close()
    
    indices = sorted(indices, key=cmp_to_key(cmp_sigma_yy))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_yy_gt = [sigma_gt_sorted[i][1] for i in range(len(label))]
    sigma_yy = [sigma_sorted[i][1] for i in range(len(label))]
    if not gt_only:
        plt.plot(data_point, sigma_yy, linewidth=1.0, label = "Sigma_yy")
    plt.plot(data_point, sigma_yy_gt, linewidth=1.0, label = "GT Sigma_yy")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_yy.png", dpi = 300)
    plt.close()

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_xy))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xy_gt = [sigma_gt_sorted[i][2] for i in range(len(label))]
    sigma_xy = [sigma_sorted[i][2] for i in range(len(label))]
    if not gt_only:
        plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_xy")
    plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_xy")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_xy.png", dpi = 300)
    plt.close()

    indices = sorted(indices, key=cmp_to_key(cmp_sigma_yx))
    sigma_gt_sorted = label[indices]
    sigma_sorted = prediction[indices]
    sigma_xy_gt = [sigma_gt_sorted[i][3] for i in range(len(label))]
    sigma_xy = [sigma_sorted[i][3] for i in range(len(label))]
    if not gt_only:
        plt.plot(data_point, sigma_xy, linewidth=1.0, label = "Sigma_yx")
    plt.plot(data_point, sigma_xy_gt, linewidth=1.0, label = "GT Sigma_yx")
    plt.legend(loc="upper left")
    plt.savefig(prefix + "_learned_sigma_yx.png", dpi = 300)
    plt.close()

def plotPotential(result_folder, n_tiling_params, tiling_params_and_strain, stress_and_potential, model, prefix = "strain_energy"):
    save_path = result_folder
    
    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params, tf.convert_to_tensor(tiling_params_and_strain), stress_and_potential, model)
    # sigma and energy are the stress and potential from the network

    elastic_potential = model(tf.convert_to_tensor(tiling_params_and_strain), training = False)

    potential_gt = stress_and_potential[:, -1] # last entry is the potential
    # potential_pred = energy.numpy() # prediction 
    potential_pred = elastic_potential.numpy() #identical to above
    indices = [i for i in range(len(potential_gt))]
    
    def compare_energy(i, j):
        return potential_gt[i] - potential_gt[j]
    indices_sorted = sorted(indices, key=cmp_to_key(compare_energy))
    print(np.max(potential_gt))
    plt.plot(indices, potential_pred[indices_sorted], linewidth=0.8, label = "prediction")
    plt.plot(indices, potential_gt[indices_sorted], linewidth=0.8, label = "GT")
    plt.legend(loc="upper right")
    plt.savefig(save_path + prefix + ".png", dpi = 300)
    plt.close()



def validate(n_tiling_params, count, model_name, validation_data, validation_label):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    model = buildNMNModel(n_tiling_params)
    model.load_weights(save_path + model_name + '.tf')

    grad_loss, e_loss, sigma, energy = testStep(n_tiling_params,validation_data, validation_label, model)
    
    plotPotential(save_path, n_tiling_params, validation_data, validation_label, model)
    plot(save_path + model_name + "_validation", sigma.numpy(), validation_label, False)

    print("validation loss grad: {} energy: {}".format(grad_loss, e_loss))


def train(n_tiling_params, model_name, train_data, train_label, validation_data, validation_label):
    batch_size = np.minimum(40000, len(train_data))
    print("batch size: {}".format(batch_size))
    
    model = buildNMNModel(n_tiling_params)
    
    
    train_vars = model.trainable_variables
    opt = Adam(learning_rate=1e-4)
    max_iter = 80000

    val_lambdasTF = tf.convert_to_tensor(validation_data)
    val_sigmasTF = tf.convert_to_tensor(validation_label)

    losses = [[], []]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # rename experiment with a counter
    count = 0
    with open('counter.txt', 'r') as f:
        count = int(f.read().splitlines()[-1])
    f = open("counter.txt", "w+")
    f.write(str(count+1))
    f.close()
    summary = Summary("./Logs/" + str(count) + "/")
    
    save_path = os.path.join(current_dir, 'Models/' + str(count) + "/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    g_norm0 = 0
    iter = 0
    log_txt = open(save_path + "/log.txt", "w+")
    # loop over epochs
    for iteration in range(max_iter):
        lambdas, sigmas = next(generator(train_data, train_label))
        if batch_size == -1:
            batch = 1
        else:
            batch = int(np.floor(len(lambdas) / batch_size))
        
        train_loss_grad = 0.0
        train_loss_e = 0.0
        g_norm_sum = 0.0
        for i in range(batch):
            mini_bacth_lambdas = lambdas[i * batch_size:(i+1) * batch_size]
            mini_bacth_sigmas = sigmas[i * batch_size:(i+1) * batch_size]

            lambdasTF = tf.convert_to_tensor(mini_bacth_lambdas)
            sigmasTF = tf.convert_to_tensor(mini_bacth_sigmas)
            
            grad, e, g_norm = trainStep(n_tiling_params, opt, lambdasTF, sigmasTF, model, train_vars)
            
            train_loss_grad += grad
            train_loss_e += e
            g_norm_sum += g_norm

        if (iteration == 0):
            g_norm0 = g_norm_sum
        validation_loss_grad, validation_loss_e, _, _ = testStep(n_tiling_params, val_lambdasTF, val_sigmasTF, model)
        
        losses[0].append(train_loss_grad + train_loss_e)
        losses[1].append(validation_loss_grad + validation_loss_e)
        
        log_txt.write("iter " + str(iteration) + " " + str(train_loss_grad.numpy()) + " " + str(train_loss_e.numpy()) + " " + str(validation_loss_grad.numpy()) + " " + str(validation_loss_e.numpy()) + " " + str(g_norm_sum.numpy()) + "\n")
        print("epoch: {}/{} train_loss_grad: {} train_loss e: {}, validation_loss_grad:{} loss_e:{} |g|: {}, |g_init|: {} ".format(iteration, max_iter, train_loss_grad, train_loss_e, \
                         validation_loss_grad, validation_loss_e, \
                        g_norm_sum, g_norm0))
        summary.saveToTensorboard(train_loss_grad, train_loss_e, validation_loss_grad, validation_loss_e, iteration)
        if iteration % 500 ==0:
            model.save_weights(save_path + model_name + '.tf')

    
    
    model.save_weights(save_path + model_name + '.tf')
    
    idx = [i for i in range(len(losses[0]))]
    plt.plot(idx, losses[0], label = "train_loss")
    plt.plot(idx, losses[1], label = "validation_loss")
    plt.legend(loc="upper left")
    plt.savefig(save_path + model_name + "_log.png", dpi = 300)
    plt.close()
    
    
if __name__ == "__main__":
    n_tiling_params = 2
    
    full_data = "../data/example_data.txt"  
    
    data_all, label_all = loadDataSplitTest(n_tiling_params, full_data, False, True)

    five_percent = int(len(data_all) * 0.05)
    
    train_data =  data_all[:-five_percent]
    train_label =  label_all[:-five_percent]

    validation_data = data_all[-five_percent:]
    validation_label = label_all[-five_percent:]

    model_name = "Example"
        
    train(n_tiling_params, model_name, 
        train_data, train_label, validation_data, validation_label)
    # validate(n_tiling_params, 334, 
    #     model_name, train_data, train_label)

    