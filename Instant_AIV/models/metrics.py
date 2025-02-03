
import numpy as np
import jax.numpy as jnp
#Errors

def relative_error(pred,exact):
    return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))

def relative_error2(pred,exact):
    return np.linalg.norm(exact-pred,2)/np.linalg.norm(exact,2)

def relative_error2_abs(pred,exact):
    return np.linalg.norm(np.abs(exact)-np.abs(pred),2)/np.linalg.norm(np.abs(exact),2)

def relative_error_0(pred,exact):
    return np.linalg.norm(exact-pred)/np.linalg.norm(exact)

def RL2_loss(residual,exact,weight=1):
    return jnp.linalg.norm(weight*residual)/jnp.linalg.norm(exact)

def MAE(pred,exact,weight=1):
    return jnp.mean(weight*jnp.abs(pred - exact))

def MSE(pred,exact,weight=1):
    return jnp.mean(weight*jnp.square(pred - exact))

def MSE_split(pred,exact,weight=1):
    return jnp.mean(weight*jnp.square(pred - exact),axis=0)

def LOG_loss(pred,exact,weight=1):
    return jnp.mean(weight*jnp.log(jnp.abs(pred - exact)+0.000001))


def normal_velocity(uvw,nxyz):
    uvw_norm=np.sum(uvw*nxyz,axis=1)[:,None]*nxyz
    return uvw_norm
    
def NLL(r,sigma,weight=1):# r is the residual r=(y-mu), sigma is the sandard deviation
    nnl=jnp.log(sigma**2)/2+(r)**2/(2*sigma**2+1e-16)
    return jnp.mean(weight*nnl)

def rms_error(pred, exact):
    return np.sqrt(np.mean((exact - pred) ** 2))

def rmeds_error(pred, exact):
    return np.sqrt(np.median((exact - pred) ** 2))

def rms_percent_error(pred, exact):
    rms = np.sqrt(np.mean((exact - pred) ** 2))
    mean_exact = np.mean(exact)
    return (rms / mean_exact) * 100

def rmeds_percent_error(pred, exact):
    rmeds = np.sqrt(np.median((exact - pred) ** 2))
    median_exact = np.median(exact)
    return (rmeds / median_exact) * 100