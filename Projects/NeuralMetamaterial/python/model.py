from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
import numpy as np
import tensorflow as tf
from tf_siren import SinusodialRepresentationDense


def get_sub_tensor(dim, start, end):
	def f(x):
		if dim == 0:
			return x[start:end]
		if dim == 1:
			return x[:, start:end]
		if dim == 2:
			return x[:, :, start:end]
		if dim == 3:
			return x[:, :, :, start:end]
	return Lambda(f)
    

def buildConstitutiveModelSiren(n_strain_entry):
    inputS = Input(shape=(n_strain_entry,),dtype=tf.float64, name="inputS")
    num_hidden = 64
    x = SinusodialRepresentationDense(num_hidden, w0=30.0, activation='sine')(inputS)
    for _ in range(5):
        x = SinusodialRepresentationDense(num_hidden, w0=1.0, activation='sine')(x)
    output = SinusodialRepresentationDense(1, w0=1.0, activation=tf.keras.activations.softplus)(x)
    model = Model(inputS, output)
    return model

##### For testing activation functions ######

def buildConstitutiveModelTanh(n_strain_entry):
    inputS = Input(shape=(n_strain_entry,),dtype=tf.float64, name="inputS")
    num_hidden = 64
    x = Dense(num_hidden, activation=tf.keras.activations.tanh)(inputS)
    for _ in range(5):
        x = Dense(num_hidden, activation=tf.keras.activations.tanh)(x)
    output = Dense(1, activation=tf.keras.activations.softplus)(x)
    model = Model(inputS, output)
    return model

def buildConstitutiveModelSwish(n_strain_entry):
    inputS = Input(shape=(n_strain_entry,),dtype=tf.float64, name="inputS")
    num_hidden = 64
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(inputS)
    for _ in range(5):
        x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    output = Dense(1, activation=tf.keras.activations.softplus)(x)
    model = Model(inputS, output)
    return model

##### NMN Model ######
def buildNMNModel(num_params, data_type=tf.float64):
    
    inputS = Input(shape=(3 + num_params,),dtype=data_type, name="inputS")
    tiling_params = get_sub_tensor(1, 0, num_params)(inputS)
    strain = get_sub_tensor(1, num_params, num_params + 3)(inputS)
    num_hidden = 256
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(tiling_params)
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    x = Dense(num_hidden, activation=tf.keras.activations.swish)(x)
    y = Dense(num_hidden, activation=tf.keras.activations.swish)(strain)
    y = Dense(num_hidden, activation=tf.keras.activations.swish)(y)
    y = Dense(num_hidden, activation=tf.keras.activations.swish)(y)
    z = Concatenate()([x, y]) 
    for i in range(5):
        z = Dense(num_hidden, activation=tf.keras.activations.swish)(z)
    output = Dense(1, activation=tf.keras.activations.softplus)(z)
    
    model = Model(inputS, output)
    return model


