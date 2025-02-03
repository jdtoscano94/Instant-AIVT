# Libraries
import numpy as np
from plots import *
import cv2
import os
import tqdm
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.tri as tri
#Dataloader
class Dataset:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, sample_function) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.sample_function = sample_function

    def __iter__(self):
        batch_idx_list = self.sample_function(self.dataset, self.batch_size)
        data = [self.dataset[batch_idx] for batch_idx in batch_idx_list]
        return iter(data)
    def __getitem__(self):
        batch_idx_list = self.sample_function(self.dataset, self.batch_size)
        data = [self.dataset[batch_idx] for batch_idx in batch_idx_list]

# Random Sampling
def random_batch(dataset: Dataset, batch_size: int):
    N = len(dataset)
    return np.split(
        np.random.permutation(N),
        np.arange(batch_size, N, batch_size),
    )
# Uniform Sampling
def get_batch(dataset: Dataset, batch_size: int):
    N = len(dataset)
    return np.split(
        np.arange(N),
        np.arange(batch_size, N, batch_size),
    )



#Rescale boundaries
def rescale_bcs(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    v_mbcs=mBCs[:,4,0:0+1].flatten()[:,None]
    u_mbcs=mBCs[:,5,0:0+1].flatten()[:,None]
    w_mbcs=mBCs[:,6,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        v_mbcs=mBCs[:,4,t_idx:t_idx+1].flatten()[:,None]
        u_mbcs=mBCs[:,5,t_idx:t_idx+1].flatten()[:,None]
        w_mbcs=mBCs[:,6,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points,u_mbcs,v_mbcs,w_mbcs))
    return mbcs_f,[dx,dy,dz]

#Rescale boundaries
def rescale_points(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points))
    return mbcs_f,[dx,dy,dz]


def normalizeZ(X,X_mean,X_std):
    H = (X- X_mean)/X_std
    return H    

def normalize_between(X,X_min,X_max,lb=-1,ub=1):
    X = (ub-lb) * (X- X_min) / (X_max - X_min)+lb 
    return X    

def identity(X,X_min,X_max):
    return X

def Encode_Fourier(X,M,N):
    t=X[0]
    x=X[1]
    y=X[2]
    P_x=2
    P_y=2
    n_num = jnp.arange(1, N+1)
    m_num = jnp.arange(1, M+1)
    n, m = jnp.meshgrid(n_num, m_num)
    n=n.flatten()
    m=m.flatten()
    w_x = 2.0 * jnp.pi / P_x
    w_y = 2.0 * jnp.pi / P_y    

    out = jnp.hstack([t,
                      x,
                      y,
                      jnp.cos(n* w_x * x)  * jnp.cos(m * w_y * y),
                      jnp.cos(n * w_x * x) * jnp.sin(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.cos(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.sin(m * w_y * y)])
    return out



def make_video(image_folder, video_name, fps):
    video_name=image_folder+str(fps)+'fps-'+video_name
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Sort the images by name
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for i in tqdm.tqdm(range(len(images))):
        image=f'{image_folder}{images[i]}'
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

def rescale_points(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points))
    return mbcs_f,[dx,dy,dz]


def extract_arrays_from_params(params_test):
    # Extract arrays from the 'params' dictionary
    params_arrays = []
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
def reconstruct_params(numpy_arrays_list, params_length, params_length_AF):
    # For ease, I'll use a pointer instead of popping items
    pointer = 0
    
    # Reconstruct the 'params' dictionary
    params_dicts = []
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


# Saving All params
def save_all_params(All_params,all_params_path):
    with h5py.File(all_params_path, 'w') as hf:
        for i, inner_list in tqdm.tqdm(enumerate(All_params)):
            group = hf.create_group(f"list_{i}")
            for j, arr in enumerate(inner_list):
                group.create_dataset(f"array_{j}", data=arr)


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


def vtu_to_npy(data="",id_data=0):
    #Choose the vtu file
    # Read the source file.
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(data)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    num_of_points = reader.GetNumberOfPoints()
    print(f"Number of Points: {num_of_points}")
    num_of_cells = reader.GetNumberOfCells()
    print(f"Number of Cells: {num_of_cells}")
    points = output.GetPoints()
    npts = points.GetNumberOfPoints()
    ## Each elemnts of x is list of 3 float [xp, yp, zp]
    x = vtk_to_numpy(points.GetData())
    print(f"Shape of point data:{x.shape}")

    ## Field value Name:
    n_arrays = reader.GetNumberOfPointArrays()
    num_of_field = 0 
    field = []
    for i in range(n_arrays):
        f = reader.GetPointArrayName(i)
        field.append(f)
        print(f"Id of Field: {i} and name:{f}")
        num_of_field += 1 
    print(f"Total Number of Field: {num_of_field}")
    u = vtk_to_numpy(output.GetPointData().GetArray(id_data))
    print(f"Shape of field: {np.shape(u)}")
    print('u: ', u.shape)
    print('x: ', x.shape)
    print(np.min(u), np.max(u))
    return x,u


def process_uneven_data(X,Y,V):
    n_x=np.unique(X).shape[0]
    n_y=np.unique(Y).shape[0]
    xi = np.linspace(np.min(X), np.max(X), n_x)
    yi = np.linspace(np.min(Y), np.max(Y), n_y)
    triang = tri.Triangulation(X, Y)
    interpolator = tri.LinearTriInterpolator(triang, V)
    x, y = np.meshgrid(xi, yi)
    Vi = interpolator(x, y)
    return x,y,Vi