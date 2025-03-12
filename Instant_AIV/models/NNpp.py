 # Libraries

import numpy as np
from jax import jit, vmap, grad
import jax.numpy as jnp
from jax.nn import sigmoid
from typing import Tuple

from typing import Tuple, List, Dict, Sequence
import h5py
from Instant_AIV.models.metrics import *
from Instant_AIV.manage.dataloader import *
import optax

#Initialization
from typing import Tuple

def glorot_normal(in_dim: int, out_dim: int) -> jnp.ndarray:
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return jnp.array(np.random.normal(loc=0.0, scale=glorot_stddev, size=(in_dim, out_dim)))


def init_params(layers: List[int], initialization_type: str = 'xavier',Network_type: str='mlp',degree: int =5,Use_ResNet: bool =False) -> dict: 
    def init_adaptive_params():
        F = 0.1 * jnp.ones(3 * len(layers) - 1)
        A = 0.1 * jnp.ones(3 * len(layers) - 1)
        return [{"a0": A[3*i], "a1": A[3*i + 1], "a2": A[3*i + 2],
                 "f0": F[3*i], "f1": F[3*i + 1], "f2": F[3*i + 2]} 
                for i in range(len(layers) - 1)]
    #Define Models:
    def init_layer_mlp(in_dim, out_dim):
        if initialization_type == 'xavier':
            W = glorot_normal(in_dim, out_dim)
        elif initialization_type == 'normal':
            W = jnp.array(np.random.normal(size=(in_dim, out_dim)))
        b = jnp.zeros(out_dim)
        g = jnp.ones(out_dim)
        return {"W": W, "b": b, "g": g}
    def init_layer_kan(in_dim, out_dim,degree=degree):
        std=1 / (in_dim * (degree + 1))
        W =jnp.array(np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim,degree+1)))
        b = jnp.zeros(out_dim)
        g = jnp.ones(out_dim)
        return {"W": W, "b": b, "g": g}
    #Select model
    if Network_type.lower()=='mlp':
            init_layer_params=init_layer_mlp
    elif Network_type.lower()[:3]=='kan':
            init_layer_params=init_layer_kan
    else:
        print(f'Error: {Network_type.lower()} is not a valid option. The available options are:mlp and kan.')
    print(f'Initializing:{Network_type} parameters.')
    params = [init_layer_params(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
    U1, b1, g1 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    U2, b2, g2 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    
    mMLP_params = [{"U1": U1, "b1": b1, "g1": g1, "U2": U2, "b2": b2, "g2": g2}]
    
    return {
        'params': params,
        'AdaptiveAF': init_adaptive_params(),
        'mMLP': mMLP_params
    }


def select_model(Network_type='mlp',degree=5):
    if Network_type.lower()=='mlp':
        return FCN
    elif Network_type.lower()=='kan':
        if degree==3:
            return KAN_Net3
        elif degree==5:
            return KAN_Net5
        elif degree==7:
            return KAN_Net7
        else:
            print('Not valid degree')
            return None

def initialize_optimizer(lr0, decay_rate, lrf, decay_step, T_e,optimizer_type='Adam',weight_decay=1e-5):
    print('Optimizer',optimizer_type.lower())
    if optimizer_type.lower()=='adam':
        if decay_rate == 0 or lrf == lr0:
            print('No decay')
            return optax.adam(lr0), decay_step
        else:
            if decay_step == 0:
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            return optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate,)),decay_step
    elif optimizer_type.lower()=='adamw':
        print('Weight decay:',weight_decay)
        if decay_rate == 0 or lrf == lr0:
            print('No decay')
            return optax.adamw(learning_rate=lr0, weight_decay=weight_decay), decay_step

        else:
            if decay_step == 0:
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            # Use adamw with the specified learning rate schedule
            return optax.adamw(optax.exponential_decay(lr0, decay_step, decay_rate), weight_decay=weight_decay), decay_step
    elif optimizer_type.lower()=='lion':
        if decay_rate == 0 or lrf == lr0:
            weight_decay=weight_decay*3
            print('No decay')
            return optax.lion(learning_rate=lr0, weight_decay=weight_decay), decay_step
        else:
            if decay_step == 0:
                weight_decay=weight_decay*3
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            # Use adamw with the specified learning rate schedule
            return optax.lion(optax.exponential_decay(lr0, decay_step, decay_rate), weight_decay=weight_decay), decay_step

def load_params_dict(result_path, dataset_name, layer_dict, initialization, type='Test',Use_ResNet=False):
    loaded_params = {}
    for key in layer_dict.keys():
        # Construct the file path
        file_path = f"{result_path}{dataset_name}-{type}_params_{key}.h5"
        
        # Read parameters from the file
        raw_params = read_all_params(file_path)[0]

        # Initialize a test parameter set for getting lengths
        test_params = init_params(layer_dict[key], initialization_type=initialization.lower()) 
        params_length = len(test_params['params'])
        params_length_AF = len(test_params['AdaptiveAF'])

        # Reconstruct the parameters
        reconstructed_params = reconstruct_params(raw_params, params_length, params_length_AF,Use_ResNet)

        # Store in the dictionary
        loaded_params[key] = reconstructed_params

    return loaded_params

def save_params_dict(params,result_path,dataset_name,type='Test',Use_ResNet=False):
    # Assuming 'result_path' and 'dataset_name' are defined elsewhere in your code
    for key, params in params.items():
        # Construct the file name for each set of parameters
        output_path = f"{result_path}{dataset_name}-{type}_params_{key}.h5"
        
        # Extract arrays from params
        params_to_save = [extract_arrays_from_params(params,Use_ResNet)]
        # Save the parameters
        save_all_params(params_to_save, output_path)
        print(f'Params {key} have been saved!')

# Reading All params
def read_all_params(all_params_path):
    loaded_All_params = []
    with h5py.File(all_params_path, 'r') as hf:
        for key in tqdm.tqdm(sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))):  # Ensure keys are processed in order
            inner_list = []
            for sub_key in sorted(hf[key].keys(), key=lambda x: int(x.split('_')[1])):
                data = hf[key][sub_key]
                if data.shape == ():  # Check if scalar
                    inner_list.append(data[()])
                else:
                    inner_list.append(data[:])
            loaded_All_params.append(inner_list)
    return loaded_All_params

def init_params_dict(layer_dict, initialization,Use_ResNet=False,Network_type='mlp',degree=5):
    print(f'You selected: Network {Network_type} with degree(if KAN) {degree}, initialization {initialization}')
    if Network_type.lower()=='mlp':
        if Use_ResNet:
            init_function=init_params_res
        else:
            init_function=init_params
    elif Network_type[:3].lower()=='kan':
            init_function=init_params
    initialized_params = {}
    for key, layer_structure in layer_dict.items():
        # Initialize parameters for each key
        params = init_function(layer_structure, 
                               initialization_type=initialization.lower(),
                               Network_type=Network_type,
                               degree=degree,
                               Use_ResNet=Use_ResNet)
        
        # Store in the dictionary
        initialized_params[key] = params

    return initialized_params

def reconstruct_params(numpy_arrays_list, params_length, params_length_AF,Use_ResNet=False):
    # For ease, I'll use a pointer instead of popping items
    pointer = 0
    # Reconstruct the 'params' dictionary
    params_dicts = []
    if Use_ResNet:
        for _ in range(params_length):
            param_dict = {}
            for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
                if pointer < len(numpy_arrays_list):
                    param_dict[key] = numpy_arrays_list[pointer]
                    pointer += 1
            params_dicts.append(param_dict)
    else:
        for _ in range(params_length):
            param_dict = {}
            for key in ['W', 'b', 'g']:
                if pointer < len(numpy_arrays_list):
                    param_dict[key] = numpy_arrays_list[pointer]
                    pointer += 1
            params_dicts.append(param_dict)

    # Reconstruct the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_dict = {}
    for key in mMLP_keys:
        if pointer < len(numpy_arrays_list):
            mMLP_dict[key] = numpy_arrays_list[pointer]
            pointer += 1

    # Reconstruct the 'AdaptiveAF' dictionary
    AdaptiveAF_dicts = []
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    for _ in range(params_length_AF):
        adaptive_dict = {}
        for key in AdaptiveAF_keys:
            if pointer < len(numpy_arrays_list):
                adaptive_dict[key] = numpy_arrays_list[pointer]
                pointer += 1
        AdaptiveAF_dicts.append(adaptive_dict)

    # Combine all reconstructed dictionaries
    reconstructed_params_test = {
        'AdaptiveAF': AdaptiveAF_dicts,
        'mMLP': [mMLP_dict],
        'params': params_dicts,
    }
    
    return reconstructed_params_test

# MLP Architecture
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

# KAN Architectures

def KAN_Net3(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer3(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer3(x,layer_params)
    return x


def KAN_Net5(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer5(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer5(x,layer_params)
    return x


def KAN_Net7(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer7(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x),
                    T6(x),
                    T7(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer7(x,layer_params)
    return x
