
import numpy as np
import jax.numpy as jnp
#Errors

def relative_error(pred,exact):
    return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))

def relative_error2(pred,exact):
    return np.linalg.norm(exact-pred,2)/np.linalg.norm(exact,2)


def relative_error_0(pred,exact):
    return np.linalg.norm(exact-pred)/np.linalg.norm(exact)

def relative_error_loss(pred,exact):
    return jnp.linalg.norm(exact-pred)/jnp.linalg.norm(exact)

def MAE(pred,exact,weight=1):
    return jnp.mean(weight*jnp.abs(pred - exact))

def MSE(pred,exact,weight=1):
    return jnp.mean(weight*jnp.square(pred - exact))

def MSE_split(pred,exact,weight=1):
    return jnp.mean(weight*jnp.square(pred - exact),axis=0)

def normal_velocity(uvw,nxyz):
    uvw_norm=np.sum(uvw*nxyz,axis=1)[:,None]*nxyz
    return uvw_norm