# Libraries

import numpy as np
from jax import jit, vmap, grad
import jax.numpy as jnp
from jax.nn import sigmoid
from typing import Tuple
from typing import List, Dict
import h5py
from metrics import *

import sys
import os
file_path = os.getcwd()
project_root = os.path.dirname(os.path.dirname(file_path))
# Add the project root to the sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
from Instant_AIV.Functions import plots, NNpp,dataloader,metrics


#Initialization
def glorot_normal(in_dim, out_dim):
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    W = glorot_stddev * jnp.array(np.random.normal(size=(in_dim, out_dim)))
    return W
def init_params(layers,type='xavier',WN=False, Mod_MLP=False):
    params = []
    in_dim, out_dim = layers[0], layers[1]
    U1=glorot_normal(in_dim, out_dim)
    b1=jnp.zeros(out_dim)
    U2=glorot_normal(in_dim, out_dim)
    b2=jnp.zeros(out_dim)
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i + 1]
        if type=='xavier':
            W = glorot_normal(in_dim, out_dim)
        elif type=='normal':
            W =jnp.array(np.random.normal(size=(in_dim, out_dim)))
        b = jnp.zeros(out_dim)
        if WN:
            g = jnp.ones([out_dim])
            if Mod_MLP and i ==0:
                g1 = jnp.ones([out_dim])
                g2 = jnp.ones([out_dim])
                params.append({"W": W, "b": b,"g":g,
                              "U1":U1, "b1":b1,"g1":g1,
                              "U2":U2, "b2":b2,"g2":g2,})
            else:
                params.append({"W": W, "b": b,"g":g})
        else:
            if Mod_MLP and i ==0:
                params.append({"W": W, "b": b,
                              "U1":U1, "b1":b1,
                              "U2":U2, "b2":b2,})
            else:
                params.append({"W": W, "b": b})
    return params



#Neural networks
def net_fn(params, X_in):
    X = X_in
    for layer in params[:-1]:
        X = jnp.sin(X @ layer["W"] + layer["b"]) 
    X = X @ params[-1]["W"] + params[-1]["b"] 
    return X
# NN with multiple activations

def FCN(params,X_in,M1,M2,activation,norm_fn):#Modified MLP
    inputs =norm_fn(X_in,M1,M2)
    for layer in params[:-1]:
        outputs = activation(inputs @ layer["W"] + layer["b"]) 
        inputs  = outputs
    W = params[-1]["W"]
    b = params[-1]["b"]
    outputs = jnp.dot(inputs, W) + b
    return outputs


def FCN_WN(params,X_in,M1,M2,activation,norm_fn):
    H =  norm_fn(X_in,M1,M2)
    for idx in range(len(params)):
        layer=params[idx]
        W=layer["W"]
        b=layer["b"]
        g=layer["g"]
        #Weight Normalization:
        V = W/jnp.linalg.norm(W, axis = 0, keepdims=True)
        #Matrix multiplication
        H = jnp.matmul(H,V)
        #Add bias
        H=g*H+b
        if idx<len(params)-1:
            H = activation(H) 
    return H

def FCN_WN_MMLP(params,X_in,M1,M2,activation,norm_fn):
    H =  norm_fn(X_in,M1,M2)
    U1=params[0]["U1"]
    U2=params[0]["U2"]
    b1=params[0]["b1"]
    b2=params[0]["b2"]
    U =activation(jnp.dot(H, U1) + b1)
    V =activation(jnp.dot(H, U2) + b2)
    for idx in range(len(params)-1):
        layer=params[idx]
        W=layer["W"]
        b=layer["b"]
        g=layer["g"]
        #Weight Normalization:
        Vn = W/jnp.linalg.norm(W, axis = 0, keepdims=True)
        #Matrix multiplication
        H = jnp.matmul(H,Vn)
        #Add bias
        H=g*H+b
        H = activation(H) 
        H  = jnp.multiply(H, U) + jnp.multiply(1 - H, V) 
    layer=params[-1]
    W=layer["W"]
    b=layer["b"]
    g=layer["g"]
    #Weight Normalization:
    Vn = W/jnp.linalg.norm(W, axis = 0, keepdims=True)
    #Matrix multiplication
    H = jnp.matmul(H,Vn)
    #Add bias
    H=g*H+b
    return H



def FCN_MMLP(params,X_in,M1,M2,activation,norm_fn):#Modified MLP
    inputs =norm_fn(X_in,M1,M2)
    # MMLP
    U1=params[0]["U1"]
    U2=params[0]["U2"]
    b1=params[0]["b1"]
    b2=params[0]["b2"]
    U =activation(jnp.dot(inputs, U1) + b1)
    V =activation(jnp.dot(inputs, U2) + b2)
    for layer in params[:-1]:
        outputs = activation(inputs @ layer["W"] + layer["b"]) 
        inputs  = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V) 
    W = params[-1]["W"]
    b = params[-1]["b"]
    outputs = jnp.dot(inputs, W) + b
    return outputs


def FCN_MMLP2(params,X_in,M1,M2,activation,norm_fn):#Modified MLP
    inputs =norm_fn(X_in,M1,M2)
    # MMLP
    U1=params[0]["U1"]
    U2=params[0]["U2"]
    b1=params[0]["b1"]
    b2=params[0]["b2"]
    U =activation(jnp.dot(inputs, U1) + b1)
    V =activation(jnp.dot(inputs, U2) + b2)
    for layer in params[:-1]:
        outputs = inputs @ layer["W"] + layer["b"]
        inputs  = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V) 
        inputs  = activation(inputs)
    W = params[-1]["W"]
    b = params[-1]["b"]
    outputs = jnp.dot(inputs, W) + b
    return outputs


# Save Resuls
def save_list(Loss,path,name='loss-'):
    filename=path+name+".npy"
    np.save(filename, np.array(Loss))
    
def save_MLP_params(params: List[Dict[str, np.ndarray]], save_path,WN=False,Mod_MLP=False):
    with h5py.File(save_path, "w") as f:
        for layer_idx, layer_params in enumerate(params):
            layer_group = f.create_group(f"Layer_{layer_idx/2.0:.2f}")
            if WN:
                if Mod_MLP:
                    W, b, g= layer_params.values()
                    layer_group.create_dataset("W", shape=W.shape, dtype=np.float32, data=W)
                    layer_group.create_dataset("b", shape=b.shape, dtype=np.float32, data=b)
                    layer_group.create_dataset("g", shape=g.shape, dtype=np.float32, data=g)
                else:
                    W, b, g= layer_params.values()
                    layer_group.create_dataset("W", shape=W.shape, dtype=np.float32, data=W)
                    layer_group.create_dataset("b", shape=b.shape, dtype=np.float32, data=b)
                    layer_group.create_dataset("g", shape=g.shape, dtype=np.float32, data=g)
            else:
                W, b = layer_params.values()
                layer_group.create_dataset("W", shape=W.shape, dtype=np.float32, data=W)
                layer_group.create_dataset("b", shape=b.shape, dtype=np.float32, data=b)  
            
def read_params(filename,WN=False):
    data = h5py.File(filename, 'r')
    recover_params=[]
    for layer in data.keys() :
        if WN:
            stored={'W':data[layer]['W'][:],'b': data[layer]['b'][:],'g': data[layer]['g'][:]}
        else:
            stored={'W':data[layer]['W'][:],'b': data[layer]['b'][:]}
        recover_params.append(stored)
    return recover_params




# PINNs

@jit
def Vorticity2D(params, X):
    # define functions
    pinn_fn         = lambda x: net_fn(params, x)
    u, v            = lambda x: pinn_fn(x)[0], lambda x: pinn_fn(x)[1]
    # compute derivatives
    v_x,  u_y,      = lambda x: grad(v)(x)[1],   lambda x: grad(u)(x)[2]   
    wz  = lambda x: v_x(x) -u_y(x)
    #Evaluate in X
    OmegaZ = vmap(wz,(0))(X)
    return OmegaZ
