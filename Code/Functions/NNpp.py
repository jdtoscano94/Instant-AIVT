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
from typing import Tuple

def glorot_normal(in_dim: int, out_dim: int) -> jnp.ndarray:
    """
    Initialize weights using Glorot (Xavier) initialization.
    
    Parameters:
    - in_dim (int): Input dimension.
    - out_dim (int): Output dimension.
    
    Returns:
    - jnp.ndarray: Initialized weights.
    """
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return jnp.array(np.random.normal(loc=0.0, scale=glorot_stddev, size=(in_dim, out_dim)))


def init_params(layers: List[int], initialization_type: str = 'xavier') -> dict:
    def init_adaptive_params():
        F = 0.1 * jnp.ones(3 * len(layers) - 1)
        A = 0.1 * jnp.ones(3 * len(layers) - 1)
        return [{"a0": A[3*i], "a1": A[3*i + 1], "a2": A[3*i + 2],
                 "f0": F[3*i], "f1": F[3*i + 1], "f2": F[3*i + 2]} 
                for i in range(len(layers) - 1)]
                
    def init_layer_params(in_dim, out_dim):
        if initialization_type == 'xavier':
            W = glorot_normal(in_dim, out_dim)
        elif initialization_type == 'normal':
            W = jnp.array(np.random.normal(size=(in_dim, out_dim)))
        b = jnp.zeros(out_dim)
        g = jnp.ones(out_dim)
        return {"W": W, "b": b, "g": g}

    params = [init_layer_params(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
    
    U1, b1, g1 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    U2, b2, g2 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    
    mMLP_params = [{"U1": U1, "b1": b1, "g1": g1, "U2": U2, "b2": b2, "g2": g2}]
    
    return {
        'params': params,
        'AdaptiveAF': init_adaptive_params(),
        'mMLP': mMLP_params
    }



#Neural networks
def net_fn(params, X_in):
    X = X_in
    for layer in params[:-1]:
        X = jnp.sin(X @ layer["W"] + layer["b"]) 
    X = X @ params[-1]["W"] + params[-1]["b"] 
    return X
# NN with multiple activations

def FCN(params, X_in, M1, M2, activation_fn, norm_fn):
    """
    Fully Connected Network (FCN) with a given normalization and activation function.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation_fn: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """
    
    params_N = params["params"]
    inputs = norm_fn(X_in, M1, M2)
    
    for layer in params_N[:-1]:
        outputs = activation_fn(jnp.dot(inputs, layer["W"]) + layer["b"])
        inputs = outputs
    
    W = params_N[-1]["W"]
    b = params_N[-1]["b"]
    outputs = jnp.dot(inputs, W) + b
    
    return outputs

def FCN_Adaptive(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with adaptive activation functions.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers and adaptive activation function coefficients.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """
    
    adaptive_coefficients = params["AdaptiveAF"]
    layers_params = params["params"]
    
    # Normalize the input
    inputs = norm_fn(X_in, M1, M2)
    
    # Iterate over all layers except the last one
    for i, (layer_params, adaptive_params) in enumerate(zip(layers_params[:-1], adaptive_coefficients)):
        pre_activation = inputs @ layer_params["W"] + layer_params["b"]
        
        # Compute the adaptive activation
        act_0 = adaptive_params["a0"] * activation(10 * adaptive_params["f0"] * pre_activation)
        act_1 = adaptive_params["a1"] * activation(20 * adaptive_params["f1"] * pre_activation)
        act_2 = adaptive_params["a2"] * activation(30 * adaptive_params["f2"] * pre_activation)
        
        inputs = 10 * (act_0 + act_1 + act_2)
    
    # For the last layer, only a linear transformation is applied
    outputs = jnp.dot(inputs, layers_params[-1]["W"]) + layers_params[-1]["b"]
    
    return outputs


def FCN_WN(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Weight Normalization.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """
    
    # Normalize the input
    H = norm_fn(X_in, M1, M2)
    
    # Iterate through the layers
    for i, layer_params in enumerate(params["params"]):
        W, b, g = layer_params["W"], layer_params["b"], layer_params["g"]
        
        # Weight normalization
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        
        # Linear transformation
        H = g * jnp.matmul(H, V) + b
        
        # Apply activation function for all layers except the last one
        if i != len(params["params"]) - 1:
            H = activation(H)
    
    return H


def FCN_WN_Adaptive(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Weight Normalization and Adaptive Activation.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers and adaptive activation parameters.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """
    
    # Normalize the input
    H = norm_fn(X_in, M1, M2)
    
    AdaptiveAF = params["AdaptiveAF"]
    params_N = params["params"]
    
    # Iterate through the layers except the last one
    for i, (layer_params, adaptive_params) in enumerate(zip(params_N[:-1], AdaptiveAF)):
        
        W, b, g = layer_params["W"], layer_params["b"], layer_params["g"]
        
        # Weight normalization
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        
        # Adaptive activation
        adaptive_act = sum([
            adaptive_params[f"a{j}"] * activation(10 * (j + 1) * adaptive_params[f"f{j}"] * (g * jnp.matmul(H, V) + b))
            for j in range(3)
        ])
        
        # Update H
        H = 10 * adaptive_act

    # For the last layer, apply only weight normalization without adaptive activation
    W, b, g = params_N[-1]["W"], params_N[-1]["b"], params_N[-1]["g"]
    V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
    H = g * jnp.matmul(H, V) + b

    return H


def FCN_WN_MMLP(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Weight Normalization and Modified MLP (MMLP) transformations.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers and MMLP parameters.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """

    # Normalize the input
    H = norm_fn(X_in, M1, M2)

    # Unpack MMLP parameters and apply weight normalization
    mMLP_params = params["mMLP"][0]
    U1, U2, b1, b2, g1, g2 = (mMLP_params[key] for key in ["U1", "U2", "b1", "b2", "g1", "g2"])
    U1_norm, U2_norm = U1 / jnp.linalg.norm(U1, axis=0, keepdims=True), U2 / jnp.linalg.norm(U2, axis=0, keepdims=True)

    # Calculate U and V transformations
    U = activation(g1 * jnp.dot(H, U1_norm) + b1)
    V = activation(g2 * jnp.dot(H, U2_norm) + b2)

    # Iterate through the layers
    for idx, layer in enumerate(params["params"][:-1]):
        W, b, g = layer["W"], layer["b"], layer["g"]

        # Apply weight normalization
        W_norm = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        
        # Compute activations and apply MMLP combination step
        H = activation(g * jnp.dot(H, W_norm) + b)
        H = jnp.multiply(H, U) + jnp.multiply(1 - H, V)

    # Process the last layer
    W, b, g = params["params"][-1]["W"], params["params"][-1]["b"], params["params"][-1]["g"]
    W_norm = W / jnp.linalg.norm(W, axis=0, keepdims=True)
    H = g * jnp.dot(H, W_norm) + b

    return H



def FCN_MMLP(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Modified MLP (MMLP) transformations.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers and MMLP parameters.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """

    # Normalize the input
    inputs = norm_fn(X_in, M1, M2)

    # Unpack MMLP parameters
    mMLP_params = params["mMLP"][0]
    U1, U2, b1, b2 = mMLP_params["U1"], mMLP_params["U2"], mMLP_params["b1"], mMLP_params["b2"]

    # Calculate U and V transformations
    U = activation(jnp.dot(inputs, U1) + b1)
    V = activation(jnp.dot(inputs, U2) + b2)

    # Iterate through all layers except the last
    for layer in params["params"][:-1]:
        W, b = layer["W"], layer["b"]
        
        # Compute activations
        act_values = activation(jnp.dot(inputs, W) + b)
        
        # MMLP combination step
        inputs = jnp.multiply(act_values, U) + jnp.multiply(1 - act_values, V)

    # Compute output from the last layer
    W, b = params["params"][-1]["W"], params["params"][-1]["b"]
    outputs = jnp.dot(inputs, W) + b

    return outputs

def FCN_MMLP_Adaptive(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Modified MLP (MMLP) transformations and Adaptive Activation functions.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers, MMLP, and adaptive activations.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """

    # Extract parameters
    AdaptiveAF = params["AdaptiveAF"]
    mMLP_params = params["mMLP"][0]
    params_N = params["params"]

    # Normalize input and extract MMLP parameters
    inputs = norm_fn(X_in, M1, M2)
    U1, U2, b1, b2 = (mMLP_params[key] for key in ["U1", "U2", "b1", "b2"])
    
    # Calculate U and V transformations
    U = activation(jnp.dot(inputs, U1) + b1)
    V = activation(jnp.dot(inputs, U2) + b2)

    # Process the layers with adaptive activation functions
    for i, layer in enumerate(params_N[:-1]):
        adapt_act = lambda factor, a, f: a * activation(factor * f * (inputs @ layer["W"] + layer["b"]))
        
        inputs = 10 * (sum(adapt_act(factor, AdaptiveAF[i][f"a{j}"], AdaptiveAF[i][f"f{j}"]) for j, factor in enumerate([10, 20, 30])))
        
        # MMLP combination step
        inputs = jnp.multiply(inputs, U) + jnp.multiply(1 - inputs, V)

    # Process the last layer
    inputs = jnp.dot(inputs, params_N[-1]["W"]) + params_N[-1]["b"]
    
    return inputs


def FCN_WN_MMLP_Adaptive(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Weight Normalization, Modified MLP (MMLP) transformations, and Adaptive Activation functions.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers, MMLP, and adaptive activations.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """

    # Extract parameters
    mMLP_params = params["mMLP"][0]
    AdaptiveAF = params["AdaptiveAF"]
    params_N = params["params"]

    # Normalize input and apply weight normalization to MMLP parameters
    H = norm_fn(X_in, M1, M2)
    U1, U2, b1, b2, g1, g2 = (mMLP_params[key] for key in ["U1", "U2", "b1", "b2", "g1", "g2"])
    U1 /= jnp.linalg.norm(U1, axis=0, keepdims=True)
    U2 /= jnp.linalg.norm(U2, axis=0, keepdims=True)

    # Calculate U and V transformations
    U = activation(g1 * jnp.matmul(H, U1) + b1)
    V = activation(g2 * jnp.matmul(H, U2) + b2)

    # Process the layers with adaptive activation functions and MMLP combination
    for idx, layer in enumerate(params_N[:-1]):
        W, b, g = layer["W"], layer["b"], layer["g"]
        
        # Weight normalization
        W /= jnp.linalg.norm(W, axis=0, keepdims=True)

        # Apply weight-normalized matrix multiplication
        H = g * jnp.matmul(H, W) + b

        # Apply adaptive activation functions
        H = 10 * sum(AdaptiveAF[idx][f"a{j}"] * activation(factor * AdaptiveAF[idx][f"f{j}"] * H) 
                     for j, factor in enumerate([10, 20, 30]))

        # MMLP combination step
        H = jnp.multiply(H, U) + jnp.multiply(1 - H, V)

    # Process the last layer with weight normalization
    W, b, g = params_N[-1]["W"], params_N[-1]["b"], params_N[-1]["g"]
    W /= jnp.linalg.norm(W, axis=0, keepdims=True)
    H = g * jnp.matmul(H, W) + b
    
    return H


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
