# Libraries

import numpy as np
from jax import jit, vmap, grad
import jax.numpy as jnp
from jax.nn import sigmoid
from typing import Tuple
from typing import List, Dict
import h5py
from metrics import *


#Initialization
def glorot_normal(in_dim, out_dim):
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    W = glorot_stddev * jnp.array(np.random.normal(size=(in_dim, out_dim)))
    return W
def init_params(layers,type='xavier',WN=False):
    params = []
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i + 1]
        if type=='xavier':
            W = glorot_normal(in_dim, out_dim)
        elif type=='normal':
            W =jnp.array(np.random.normal(size=(in_dim, out_dim)))
        b = jnp.zeros(out_dim)
        if WN:
            g = jnp.ones([out_dim])
            params.append({"W": W, "b": b,"g":g})
        else:
            params.append({"W": W, "b": b})
    return params

def init_MMLP(layers,type='xavier',WN=False):
    params = []
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i + 1]
        if type=='xavier':
            W = glorot_normal(in_dim, out_dim)
        elif type=='normal':
            W =jnp.array(np.random.normal(size=(in_dim, out_dim)))
        b = jnp.zeros(out_dim)
        if WN:
            g = jnp.ones([out_dim])
            params.append({"W": W, "b": b,"g":g})
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

def FCN(params,X_in,M1,M2,activation,norm_fn):
    X =norm_fn(X_in,M1,M2)
    for layer in params[:-1]:
        X = activation(X @ layer["W"] + layer["b"]) 
    X = X @ params[-1]["W"] + params[-1]["b"] 
    return X


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



# Save Resuls
def save_list(Loss,path,name='loss-'):
    filename=path+name+".npy"
    np.save(filename, np.array(Loss))
    
def save_MLP_params(params: List[Dict[str, np.ndarray]], save_path,WN=False):
    with h5py.File(save_path, "w") as f:
        for layer_idx, layer_params in enumerate(params):
            layer_group = f.create_group(f"Layer_{layer_idx/2.0:.2f}")
            if WN:
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




# Further 
@jit
def predict_uvwp(params, X_in) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uvwp = net_fn(params, X_in)
    u, v, w, p = uvwp[:, 0], uvwp[:, 1], uvwp[:, 2], uvwp[:, 3]
    return u, v, w,p


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
