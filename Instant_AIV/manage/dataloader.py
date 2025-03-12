# Libraries
import numpy as np
from Instant_AIV.manage.plots import *
from Instant_AIV.models.metrics import *
import cv2
import os
import tqdm
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.tri as tri
import jax.numpy as jnp
import jax


    
def create_and_return_directories(save_path, dataset_name, subdirectories):
    # Base directory
    result_path = os.path.join(save_path, dataset_name)

    # Creating subdirectories and storing their paths
    paths = {}
    for subdir in subdirectories:
        path = os.path.join(result_path, subdir+'/')
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path

        # Printing the paths
        print(f"The {subdir.lower().replace('_', ' ')} path is: {path}")

    return paths


def normalizeZ(X,X_mean,X_std):
    H = (X- X_mean)/X_std
    return H    

def normalize_between(X,X_min,X_max,lb=-1,ub=1):
    X = (ub-lb) * (X- X_min) / (X_max - X_min)+lb 
    return X    

def identity(X,X_min,X_max):
    return X


          
def sample_points_PDF(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=lambdas_key/lambdas_key.sum())
    return batch_indices

   
# Activation functions mapping
ACTIVATION_FUNCTIONS = {
    'sin': jnp.sin,
    'tanh': jnp.tanh,
    'tanh_08': lambda x: 0.8*jnp.tanh(x),
    'swish': lambda x: x * jax.nn.sigmoid(x),
    'leaky_relu': lambda x: jnp.where(x > 0, x, 0.01 * x),
    'custom': lambda x: jnp.where(x < 0, -1, jnp.where(x > 2, 1, 0.5*(x)**2-1)),
    'sigmoid':lambda x: jax.nn.sigmoid(x),
    'sigmoid_11':lambda x: 2/(1+jnp.exp(-x*1.1**2))-1

}

# Normalization functions and metrics mapping
NORMALIZATION_FUNCTIONS = {
    'plusminus1': {
        'fn': normalize_between,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'normal': {
        'fn': normalizeZ,
        'metric1': lambda x: x.mean(0, keepdims=False),
        'metric2': lambda x: x.std(0, keepdims=False)
    },
    'identity': {
        'fn': identity,
        'metric1': lambda x: 0,
        'metric2': lambda x: 0
    }
}

# Loss metric mapping
ERROR_FUNCTIONS = {
    'l2': MSE,
    'l1': MAE
}


def filter_Magnitude(BCs_frame,row=7,T_max=0.7,T_min=0.49):
    T  =BCs_frame[:,row]
    upper_limit = T_max
    lower_limit = T_min
    idx1=np.argwhere(T<upper_limit)
    idx2=np.argwhere(T>lower_limit)
    idxT=np.intersect1d(idx1,idx2)
    BCs_framef=BCs_frame[idxT]
    return BCs_framef

def filter_Z(BCs_frame,row=7,permissibility=3):
    u  =BCs_frame[:,row]
    #Z score FILTERING
    #Filter u
    mean_u = np.nanmean(u)
    std_u  = np.nanstd(u)
    upper_limit = mean_u + permissibility*std_u
    lower_limit = mean_u - permissibility*std_u
    idx1=np.argwhere(u<upper_limit)
    idx2=np.argwhere(u>lower_limit)
    idx=np.intersect1d(idx1,idx2)
    BCs_framef=BCs_frame[idx]
    return BCs_framef

# Chebyshev's Polynomials
def T0(x):
    return x*0+1
def T1(x):
    return x
def T2(x):
    return 2*x**2-1
def T3(x):
    return 4*x**3-3*x
def T4(x):
    return 8*x**4-8*x**2+1
def T5(x):
    return 16*x**5-20*x**3+5*x
def T6(x):
    return 32*x**6-48*x**4+18*x**2-1
def T7(x):
    return 64*x**7-112*x**5+56*x**3-7*x



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

def extract_arrays_from_params(params_test,Use_ResNet=False):
    # Extract arrays from the 'params' dictionary
    params_arrays = []
    if Use_ResNet:
        for param_dict in params_test['params']:
            for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
                if key in param_dict:
                    params_arrays.append(np.array(param_dict[key]))
    else:
        for param_dict in params_test['params']:
            for key in ['W', 'b', 'g']:
                if key in param_dict:
                    params_arrays.append(np.array(param_dict[key]))

    # Extract arrays from the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_arrays = [np.array(params_test['mMLP'][0][key]) for key in mMLP_keys if key in params_test['mMLP'][0]]

    # Extract arrays from the 'AdaptiveAF' dictionary
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    AdaptiveAF_arrays = []
    for adaptive_dict in params_test['AdaptiveAF']:
        for key in AdaptiveAF_keys:
            AdaptiveAF_arrays.append(np.array(adaptive_dict[key]))

    # Combine all extracted arrays
    all_arrays = params_arrays + mMLP_arrays + AdaptiveAF_arrays
    return all_arrays

# Saving All params
def save_all_params(All_params,all_params_path):
    with h5py.File(all_params_path, 'w') as hf:
        for i, inner_list in tqdm.tqdm(enumerate(All_params)):
            group = hf.create_group(f"list_{i}")
            for j, arr in enumerate(inner_list):
                group.create_dataset(f"array_{j}", data=arr)