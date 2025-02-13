# Libraries
import numpy as np
import matplotlib.pyplot as plt
import math as m
import math
import numpy as np
from typing import List, Dict
import matplotlib.tri as tri
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.decomposition import PCA
from matplotlib import cm
# Plots
def plot(points,show=False,mark=1):
  input=points
  fig = plt.figure()
  ax_x = fig.add_subplot(111, projection='3d')
  x_ = input
  l_min=np.min(x_.flatten())
  l_max=np.max(x_.flatten())
  ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2],s=mark, marker='o')
  ax_x.set_xlim([l_min,l_max])
  ax_x.set_ylim([l_min,l_max])
  ax_x.set_zlim([l_min,l_max])
  plt.axis('off')
  if show==True:
    ax_x.view_init(azim=1, elev=90)
    plt.axis('off')
  
def plotAIV3D(full_data,f_names,plot_type='mid_z',hor_size=6,window=4,cmap='RdBu_r',size=0.1,font_size='14',save_path=''):
    #Domain
    t_test, x_test, y_test, z_test  =full_data[:,0:1],full_data[:,1:2],full_data[:,2:3],full_data[:,3:4]
    tplot=t_test[0][0]
    midplane_x=0.5*(np.max(x_test)+np.min(x_test))
    midplane_y=0.5*(np.max(y_test)+np.min(y_test))
    midplane_z=0.5*(np.max(z_test)+np.min(z_test))
    if plot_type=='mid_y':
        idx_plot=np.argwhere(y_test>=midplane_y)[:,0]
    elif plot_type=='mid_x':
        idx_plot=np.argwhere(x_test<=midplane_x)[:,0]
    elif plot_type=='mid_z':
        idx_plot=np.argwhere(z_test<=midplane_z)[:,0]
    else:
        idx_plot=np.arange(len(y_test)) 
    #File name:
    image_name=save_path+f't={tplot:.3f}s-{plot_type}.png'
    #Functions
    F=full_data[:,full_data.shape[1]-len(f_names)+1:]
    #Compute velocity
    V_pred=np.sqrt(F[:,0]**2+F[:,1]**2+F[:,2]**2)[:,None]
    F=np.hstack((F,V_pred))
    F=F.T
    #Plot
    lower_lim=np.min(full_data[:,1:4],axis=0)
    upper_lim=np.max(full_data[:,1:4],axis=0)
    fig = plt.figure(figsize=(hor_size*len(F),window))
    plt.rcParams['font.size'] = font_size
    for i in range(len(F)):
        ax = fig.add_subplot(1, len(F), i+1, projection='3d')
        img = ax.scatter(x_test[idx_plot], y_test[idx_plot], z_test[idx_plot], c=F[i,idx_plot], cmap=cmap,s=size)
        fig.colorbar(img,fraction=0.046, pad=0.04)
        ax.set_title(f'{f_names[i]}(t=({tplot:.3f})({plot_type})')
        ax.set_xlim([lower_lim[0],upper_lim[0]])
        ax.set_ylim([lower_lim[1],upper_lim[1]])
        ax.set_zlim([lower_lim[2],upper_lim[2]])
        plt.axis('off')
    plt.savefig(image_name)


def plot3D(x,y,F,f_names,window=4,tplot=0,zplot=0,hor_size=4,cmap="rainbow",num_col=50,vmin=-1,vmax=1):
  X,Y= x[tplot,:,:,zplot],y[tplot,:,:,zplot]
  fig,ax=plt.subplots(1,len(F),figsize=(hor_size*len(F),window))
  for i in range(len(ax)):
    cp = ax[i].contourf(X,Y, F[i,tplot,:,:,zplot],num_col,cmap=cmap)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax[i].set_title(f_names[i]+' for (t,z)=('+str(tplot)+','+str(zplot)+')')
    ax[i].set_xlabel('x')
    if i==0:
        ax[i].set_ylabel('y')

def plot3D_mat(x,y,F,f_names,window=4,font_size='10',x_label='x',y_label='y',cmap='rainbow',fig_width=6):
  X,Y= x,y
  fig,ax=plt.subplots(1,len(F),figsize=(fig_width*len(F),window))
  plt.rcParams['font.size'] = font_size
  for i in range(len(ax)):
    cp = ax[i].contourf(X,Y, F[i],50,cmap=cmap)
    fig.colorbar(cp) # Add a colorbar to a plot
    ax[i].set_title(f_names[i])
    ax[i].set_xlabel(x_label)
    if i==0:
        ax[i].set_ylabel(y_label)
    
def plot_views(points,plot_size=(15,9),marker_size=0.008,color=None,hide_axis=True):
    x_b=points[:,0]
    y_b=points[:,1]
    z_b=points[:,2]
    
    #Isometric
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    if hide_axis:
        plt.axis('off')

    #Top
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    ax.view_init(azim=1, elev=90)
    if hide_axis:
        plt.axis('off')


    #Bottom
    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    ax.view_init(azim=1, elev=270)
    if hide_axis:
        plt.axis('off')


    #Side
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    ax.view_init(azim=1, elev=180)
    if hide_axis:
        plt.axis('off')


    #Front
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.scatter(y_b, x_b, z_b,s=marker_size, marker='o',c=color)
    ax.view_init(azim=1, elev=0)
    if hide_axis:
        plt.axis('off')

    #back
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.scatter(y_b, x_b, z_b,s=marker_size, marker='o',c=color)
    ax.view_init(azim=1, elev=180)
    if hide_axis:
        plt.axis('off')



    plt.show()


def get_txyz_at(X_plot,var='t',t_id=0,type='slice'):
    t_plot=X_plot[:,0:1]
    x_plot=X_plot[:,1:2]
    y_plot=X_plot[:,2:3]
    z_plot=X_plot[:,3:4]
    if type=='slice':
        if var =='t':
            idx_t=np.argwhere(t_plot.flatten()==np.unique(t_plot)[t_id])
        elif var =='x':
            idx_t=np.argwhere(x_plot.flatten()==np.unique(x_plot)[t_id])
        elif var =='y':
            idx_t=np.argwhere(y_plot.flatten()==np.unique(y_plot)[t_id])
        elif var =='z':
            idx_t=np.argwhere(z_plot.flatten()==np.unique(z_plot)[t_id])
    elif type=='low':
        if var =='t':
            idx_t=np.argwhere(t_plot.flatten()<=np.unique(t_plot)[t_id])
        elif var =='x':
            idx_t=np.argwhere(x_plot.flatten()<=np.unique(x_plot)[t_id])
        elif var =='y':
            idx_t=np.argwhere(y_plot.flatten()<=np.unique(y_plot)[t_id])
        elif var =='z':
            idx_t=np.argwhere(z_plot.flatten()<=np.unique(z_plot)[t_id])
    elif type=='up':
        if var =='t':
            idx_t=np.argwhere(t_plot.flatten()>=np.unique(t_plot)[t_id])
        elif var =='x':
            idx_t=np.argwhere(x_plot.flatten()>=np.unique(x_plot)[t_id])
        elif var =='y':
            idx_t=np.argwhere(y_plot.flatten()>=np.unique(y_plot)[t_id])
        elif var =='z':
            idx_t=np.argwhere(z_plot.flatten()>=np.unique(z_plot)[t_id])
    t=t_plot.flatten()[idx_t]
    x=x_plot.flatten()[idx_t]
    y=y_plot.flatten()[idx_t]
    z=z_plot.flatten()[idx_t]
    return(t,x,y,z)
    
def plot_uvwp_at(X,idz,idt,figsize=(12,7),type='slice',fontsize='10',p_scale=100,t_f=None,params=None,pinn_fn=None,save_img=True,save_path='',lamB=100,lamD=50,lamE=1,recover_units=False):
    lower_lim=np.min(X,axis=0)
    upper_lim=np.max(X,axis=0)
    _,x,y,z=get_txyz_at(X,var='z',t_id=idz,type=type)
    t=np.ones(np.shape(x))*t_f[:,idt]
    X_plot=np.hstack((t,x,y,z))
    uvwp_plot  = pinn_fn(params,X_plot)
    u=uvwp_plot[:,0:1]
    v=uvwp_plot[:,1:2]
    w=uvwp_plot[:,2:3]
    p=uvwp_plot[:,3:4]*p_scale
    V=np.sqrt(u**2+v**2+w**2)
    
    #recover units
    delta_t = 1/29.16       # time interval between 2 snapshots - s
    delta_x = 0.648e-6      # space interval between 2 pixel - m
    delta_z = 0.648e-6
    unit_x = delta_x       # space unit - 1 pixel = unit_x m
    unit_z = delta_z        # space unit - 1 pixel = unit_z m
    unit_u = delta_x/delta_t # velocity unit - 1 ppf = unit_u m/s
    # Parameters used for non-dimensionalization
    L_char = 90*unit_x          # meter (90 pixel)
    U_char = 10*unit_u          # m/s  (10 pixel/frame)
    visco = 0.7e-6              # kinematic viscocity m^2/s
    T_char = L_char / U_char     
    ReyNum = U_char*L_char / visco
    #Scale time
    T_cycle=0.3036      
    if recover_units:
        
        t=t*T_char     #seconds
        x=x*L_char*1e6 #um -> micrometers
        y=y*L_char*1e6
        z=z*L_char*1e6
        lower_lim=lower_lim*L_char*1e6
        upper_lim=upper_lim*L_char*1e6
        
        u=u*U_char*1e6 #um/s
        v=v*U_char*1e6
        w=w*U_char*1e6
        V=V*U_char*1e6
        p=p*(U_char**2)*993 #Pa
    #plot V
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = '10'
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    img = ax.scatter(x, y, z, c=V, cmap='jet',s=5)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([lower_lim[1],upper_lim[1]])
    ax.set_ylim([lower_lim[2],upper_lim[2]])
    ax.set_zlim([lower_lim[3],upper_lim[2]])
    if type=='slice':
        ax.set_title(f'V(t={t[0][0]:.3f},x,y,z={np.max(z):.3f})',fontsize=fontsize)
        if recover_units:
            ax.set_title(f'V(t={t[0][0]:.3f}s,x,y,z={np.max(z):.3f}um)[um/s]',fontsize=fontsize)
    elif type=='low':
        ax.set_title(f'V(t={t[0][0]:.3f},x,y,z<{np.max(z):.3f})',fontsize=fontsize)
        if recover_units:
            ax.set_title(f'V(t={t[0][0]:.3f}s,x,y,z<{np.max(z):.3f}um)[um/s]',fontsize=fontsize)
    plt.axis('off')
    #plot P
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    img = ax.scatter(x, y, z, c=p, cmap='jet',s=5)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([lower_lim[1],upper_lim[1]])
    ax.set_ylim([lower_lim[2],upper_lim[2]])
    ax.set_zlim([lower_lim[3],upper_lim[2]])
    if type=='slice':
        ax.set_title(f'P(t={t[0][0]:.3f},x,y,z={np.max(z):.3f})',fontsize=fontsize)
        if recover_units:
            ax.set_title(f'P(t={t[0][0]:.3f}s,x,y,z={np.max(z):.3f}um)[Pa]',fontsize=fontsize)
    elif type=='low':
        ax.set_title(f'P(t={t[0][0]:.3f},x,y,z<{np.max(z):.3f})',fontsize=fontsize)
        if recover_units:
            ax.set_title(f'P(t={t[0][0]:.3f}s,x,y,z<{np.max(z):.3f}um)[Pa]',fontsize=fontsize)
    plt.axis('off')
    if save_img:
        plt.savefig(save_path+f'/{lamB,lamD,lamE}at_t={idt},z={idz}.png')
    plt.show()


# Move and align Data
def centroid(arrdwn):
  cx=0
  cy=0
  cz=0
  for i in range(len(arrdwn)):
    cx=cx+arrdwn[i][0]
    cy=cy+arrdwn[i][1]
    cz=cz+arrdwn[i][2]
  cx=cx/len(arrdwn)
  cy=cy/len(arrdwn)
  cz=cz/len(arrdwn)
  print("the centroid is at:",cx,cy,cz)
  return cx,cy,cz

def MoveScale3D(points,dx=0,dy=0,dz=0,sx=1,sy=1,sz=1):
  other=points[:,3:]
  points=points[:,:3]
  T=np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[dx,dy,dz,1]])
  aux=np.ones((points.shape[0],points.shape[1]+1))
  aux[:,:-1] = points
  points=aux@T
  points=points[:,[0,1,2]]
  points=np.hstack((points,other))
  return points

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
fac=180/np.pi

def angles(a,graph=False):
  fac=180/np.pi
  pca=PCA(n_components=3)
  pca.fit(a)
  eigen_vectors=pca.components_
  theta1=np.array(np.arccos(eigen_vectors))
  tx=np.pi-theta1[0][0]
  ty=np.pi/2-theta1[0][1]
  tz=np.pi/2-theta1[0][2]
  origin = np.array([[0, 0, 0],[0, 0, 0]]) # origin point
  if (graph==True):
    plt.quiver(*origin, eigen_vectors[:,0], eigen_vectors[:,1], color=['r','b','g'], scale=3)
    plt.show()
    plt.quiver(*origin, eigen_vectors[:,1], eigen_vectors[:,2], color=['r','b','g'], scale=3)
    plt.show()
    plt.quiver(*origin, eigen_vectors[:,0], eigen_vectors[:,2], color=['r','b','g'], scale=3)
    plt.show()
  return tx,ty,tz
fac=180/np.pi

def plot_views_2geo(points,points2,plot_size=(15,9),marker_size=0.008,color=None,color2=None,hide_axis=True):
    x_b=points[:,0]
    y_b=points[:,1]
    z_b=points[:,2]
    x_b2=points2[:,0]
    y_b2=points2[:,1]
    z_b2=points2[:,2]
    
    #Isometric
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    ax.scatter(x_b2, y_b2, z_b2,s=marker_size, marker='o',c=color2)
    if hide_axis:
        plt.axis('off')

    #Top
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    ax.scatter(x_b2, y_b2, z_b2,s=marker_size, marker='o',c=color2)
    ax.view_init(azim=1, elev=90)
    if hide_axis:
        plt.axis('off')


    #Bottom
    ax = fig.add_subplot(2, 3, 3, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    ax.scatter(x_b2, y_b2, z_b2,s=marker_size, marker='o',c=color2)
    ax.view_init(azim=1, elev=270)
    if hide_axis:
        plt.axis('off')


    #Side
    ax = fig.add_subplot(2, 3, 4, projection='3d')
    ax.scatter(x_b, y_b, z_b,s=marker_size, marker='o',c=color)
    ax.scatter(x_b2, y_b2, z_b2,s=marker_size, marker='o',c=color2)
    ax.view_init(azim=1, elev=180)
    if hide_axis:
        plt.axis('off')


    #Front
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    ax.scatter(y_b, x_b, z_b,s=marker_size, marker='o',c=color)
    ax.scatter(y_b2, x_b2, z_b2,s=marker_size, marker='o',c=color2)
    ax.view_init(azim=1, elev=0)
    if hide_axis:
        plt.axis('off')

    #back
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.scatter(y_b, x_b, z_b,s=marker_size, marker='o',c=color)
    ax.scatter(y_b2, x_b2, z_b2,s=marker_size, marker='o',c=color2)
    ax.view_init(azim=1, elev=180)
    if hide_axis:
        plt.axis('off')
    plt.show()
    

# Fish

def plot_fish_frame(frame,X_data_frame,u_pred, v_pred, p_pred,wz_pred,V_pred,dataset_name,images_path):
    time_plot=frame
    x_test  = X_data_frame[:,1:2]
    y_test  = X_data_frame[:,2:3]
    
    u_test  = X_data_frame[:,3:4]
    v_test  = X_data_frame[:,4:5]
    p_test  = X_data_frame[:,5:6]
    wz_test = X_data_frame[:,6:7]
    V_test  = X_data_frame[:,7:8]
    
    size=0.1
    figsize=(20,6)
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = '10'
    ax = fig.add_subplot(2, 5, 1)
    img = ax.scatter(x_test, y_test, c=V_test, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'V(Test) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 2)
    img = ax.scatter(x_test, y_test, c=u_test, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'u(Test) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 3)
    img = ax.scatter(x_test, y_test, c=v_test, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'v(Test) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 4)
    img = ax.scatter(x_test, y_test, c=p_test, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'p(Test) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 5)
    img = ax.scatter(x_test, y_test, c=wz_test, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'wz(Test) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 6)
    img = ax.scatter(x_test, y_test, c=V_pred, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'V(pred) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 7)
    img = ax.scatter(x_test, y_test, c=u_pred, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'u(pred) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 8)
    img = ax.scatter(x_test, y_test, c=v_pred, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'v(pred) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 9)
    img = ax.scatter(x_test, y_test, c=p_pred, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'p(pred) at: {time_plot}T')
    plt.axis('off')
    ax = fig.add_subplot(2, 5, 10)
    img = ax.scatter(x_test, y_test, c=wz_pred, cmap='RdBu_r',s=size)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_title(f'wz(pred) at: {time_plot}T')
    plt.axis('off')
    plt.savefig(images_path+f'/{dataset_name}-Results-at:{time_plot:.2f}.png')
    
    
def plot_frame(points,frame,font_size='14',figsize=(16,7),hide_axis=True,cmap='YlOrBr',plot_frame=False,save_fig=False,save_path=""):
    fig_name=save_path+'Frame:00'+str(frame)+'.png'
    t_plot=points[:,0:1]
    x_plot=points[:,1:2]
    y_plot=points[:,2:3]
    v_plot=points[:,3:4]
    u_plot=points[:,4:5]
    idx_t=np.argwhere(t_plot.flatten()==np.unique(t_plot)[frame])
    t=t_plot.flatten()[idx_t]
    x=x_plot.flatten()[idx_t]
    y=y_plot.flatten()[idx_t]
    u=u_plot.flatten()[idx_t]
    v=v_plot.flatten()[idx_t]
    v=np.sqrt(u**2+v**2)
    if plot_frame:
        frame_plot=points[:,5:6] #frame idx
        u=frame_plot.flatten()[idx_t]
    #print(f'Number of points: {len(t)}')
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = font_size
    ax = fig.add_subplot(1, 2, 1)
    img = ax.scatter(x, y, c=u, cmap=cmap,s=5)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_title('vx')
    if hide_axis:
        plt.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    img = ax.scatter(x, y, c=v, cmap=cmap,s=5)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_title('V')
    if hide_axis:
        plt.axis('off')
    if save_fig:
        plt.savefig(fig_name)
  

    
def plot_frame_bw(points,frame,font_size='14',figsize=(16,7),n_frames=110,hide_axis=True,cmap='YlOrBr',plot_frame=False):
    t_plot=points[:,0:1]
    x_plot=points[:,1:2]
    y_plot=points[:,2:3]
    v_plot=points[:,1:2]*0+1
    u_plot=points[:,2:3]*0+1
    idx_t0=np.argwhere(t_plot.flatten()>=(frame)*np.max(t_plot)/n_frames)
    idx_t1=np.argwhere(t_plot.flatten()<(frame+1)*np.max(t_plot)/n_frames)
    idx_t=np.intersect1d(idx_t0,idx_t1)
    t=t_plot.flatten()[idx_t]
    x=x_plot.flatten()[idx_t]
    y=y_plot.flatten()[idx_t]
    u=u_plot.flatten()[idx_t]
    v=v_plot.flatten()[idx_t]
    v=np.sqrt(u**2+v**2)
    if plot_frame:
        frame_plot=points[:,3:4] #frame idx
        u=frame_plot.flatten()[idx_t]
    print(f'Number of points: {len(t)}')
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = font_size
    ax = fig.add_subplot(1, 2, 1)
    img = ax.scatter(x, y, c=u, cmap=cmap,s=5)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if hide_axis:
        plt.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    img = ax.scatter(x, y, c=v, cmap=cmap,s=5)
    fig.colorbar(img,fraction=0.046, pad=0.04)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if hide_axis:
        plt.axis('off')
  

#NSFNETSc
def plot_cNSFNet2D(full_data,f_names,hor_size=6,window=4,cmap='RdBu_r',size=0.1,font_size='14',save_path=''):
        #Domain
    t_test, x_test, y_test =full_data[:,0:1],full_data[:,1:2],full_data[:,2:3]
    tplot=t_test[0][0]
    #File name:
    image_name=save_path+f't={tplot:.3f}s.png'
    #Functions
    F=full_data[:,full_data.shape[1]-len(f_names):]
    F=F.T
    #Plot
    fig = plt.figure(figsize=(hor_size*len(F),window))
    plt.rcParams['font.size'] = font_size
    for i in range(len(F)):
        ax = fig.add_subplot(1, len(F), i+1)
        img =ax.scatter(x_test, y_test, c=F[i], cmap=cmap,s=size)
        fig.colorbar(img,fraction=0.046, pad=0.04)
        ax.set_title(f'{f_names[i]}(t=({tplot:.3f}))')
        plt.axis('off')
    plt.savefig(image_name)


def get_pdf(u_hist,bins=30):
    hist,bins=np.histogram(u_hist,bins=bins,density=True)
    prob_density=hist
    return prob_density,bins

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


def plot_losses_grid(log_loss,num_cols=3,fig_h=16,fig_v=12):
    
    titles = list(log_loss[0].keys())
    
    # Make sure the subplot grid dimensions match the number of titles
    num_rows = (len(titles) + 2) // 3  # Add 2 to round up
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_h, fig_v))

    # Ensure axs is a 2D array
    if num_rows == 1:
        axs = np.array([axs])


    plot_its=0
    time_all=0
    it_all=0
    for i, title in enumerate(titles):
        if title=='it':
            it_all = [entry[title] for entry in log_loss]
            plot_its=1
        if title=='time':
            time_all = [entry[title] for entry in log_loss]

    for i, title in enumerate(titles):
        row = i // 3
        col = i % 3
        ax = axs[row][col]

        # Extract values for a specific loss from all dictionary entries
        loss_values = [entry[title] for entry in log_loss]

        if plot_its:
            ax.plot(it_all,loss_values, label=title, color='k')
        else:
            ax.plot(loss_values, label=title, color='k')
        ax.set_title(title)
        ax.set_yscale('log')  # set y-axis to log scale
        ax.grid(True, which="both", ls="--", c='0.65')

    for ax in axs[-1, :]:
        ax.set_xlabel('Iterations (10e2)')

    plt.tight_layout()
    if plot_its:
        return it_all,time_all




def get_pdf(u_hist,bins=30):
    hist,bins=np.histogram(u_hist,bins=bins,density=True)
    prob_density=hist
    return prob_density,bins
def get_colors_plot(cmap='RdBu',n_colors=2):
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
    return colors




def np_to_vtp(surf_file, x, y, z, surf_name='train_loss', log=True, zmax=-1, interp=1):
    # set this to True to generate points
    show_points = False
    # set this to True to generate polygons
    show_polys = True

    scale = 5.0
    x = x * scale
    y = y * scale

    [xcoordinates, ycoordinates] = np.meshgrid(x[:], y[:])
    vals = z

    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals[:].ravel()

    if interp > 0:
        m = interpolate.interp2d(xcoordinates[0, :], ycoordinates[:, 0], vals, kind='cubic')
        x_array = np.linspace(min(x_array), max(x_array), interp)
        y_array = np.linspace(min(y_array), max(y_array), interp)
        z_array = m(x_array, y_array).ravel()

        x_array, y_array = np.meshgrid(x_array, y_array)
        x_array = x_array.ravel()
        y_array = y_array.ravel()

    vtp_file = surf_file + "_" + surf_name
    if zmax > 0:
        z_array[z_array > zmax] = zmax
        vtp_file += "_zmax=" + str(zmax)

    if log:
        z_array = np.log(z_array + 0.1)
        vtp_file += "_log"
    vtp_file += ".vtp"
    print("Here's your output file:{}".format(vtp_file))

    number_points = len(z_array)
    print("number_points = {} points".format(number_points))

    matrix_size = int(math.sqrt(number_points))
    print("matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    print("poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    print("number_polys = {}".format(number_polys))

    min_value_array = [min(x_array), min(y_array), min(z_array)]
    max_value_array = [max(x_array), max(y_array), max(z_array)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    poly_count = 0
    for column_count in range(poly_size):
        stride_value = column_count * matrix_size
        for row_count in range(poly_size):
            temp_index = stride_value + row_count
            averaged_z_value = (z_array[temp_index] + z_array[temp_index + 1] +
                                z_array[temp_index + matrix_size] +
                                z_array[temp_index + matrix_size + 1]) / 4.0
            averaged_z_value_array.append(averaged_z_value)
            poly_count += 1

    print(f"Avg val: {averaged_z_value_array}")

    avg_min_value = min(averaged_z_value_array)
    avg_max_value = max(averaged_z_value_array)

    output_file = open(vtp_file, 'w')
    output_file.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    output_file.write('  <PolyData>\n')

    if (show_points and show_polys):
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(
                number_points, number_points, number_polys))
    elif (show_polys):
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(
                number_points, number_polys))
    else:
        output_file.write(
            '    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(
                number_points, number_points))

    # <PointData>
    output_file.write('      <PointData>\n')
    output_file.write(
        '        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
            min_value_array[2], max_value_array[2]))
    for vertexcount in range(number_points):
        if (vertexcount % 6) == 0:
            output_file.write('          ')
        output_file.write('{}'.format(z_array[vertexcount]))
        if (vertexcount % 6) == 5:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 6) != 5:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </PointData>\n')

    # <CellData>
    output_file.write('      <CellData>\n')
    if (show_polys and not show_points):
        output_file.write(
            '        <DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
                avg_min_value, avg_max_value))
        for vertexcount in range(number_polys):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
        output_file.write('        </DataArray>\n')
    output_file.write('      </CellData>\n')

    # <Points>
    output_file.write('      <Points>\n')
    output_file.write(
        '        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(
            min_value, max_value))
    for vertexcount in range(number_points):
        if (vertexcount % 2) == 0:
            output_file.write('          ')
        output_file.write('{} {} {}'.format(x_array[vertexcount], y_array[vertexcount], z_array[vertexcount]))
        if (vertexcount % 2) == 1:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 2) != 1:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Points>\n')

    # <Verts>
    output_file.write('      <Verts>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_points - 1))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_points))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount + 1))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Verts>\n')

    # <Lines>
    output_file.write('      <Lines>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Lines>\n')

    # <Strips>
    output_file.write('      <Strips>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Strips>\n')

    # <Polys>
    output_file.write('      <Polys>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(
            number_polys - 1))
    if (show_polys):
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) == 0:
                    output_file.write('          ')
                output_file.write('{} {} {} {}'.format(temp_index, (temp_index + 1), (temp_index + matrix_size + 1),
                                                       (temp_index + matrix_size)))
                if (polycount % 2) == 1:
                    output_file.write('\n')
                else:
                    output_file.write(' ')
                polycount += 1
        if (polycount % 2) == 1:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write(
        '        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(
            number_polys))
    if (show_polys):
        for polycount in range(number_polys):
            if (polycount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format((polycount + 1) * 4))
            if (polycount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (polycount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Polys>\n')

    output_file.write('    </Piece>\n')
    output_file.write('  </PolyData>\n')
    output_file.write('</VTKFile>\n')
    output_file.write('')
    output_file.close()

    print("Done with file:{}".format(vtp_file))


# if __name__ == '__main__':
#     nu = 0.01
#     x = np.load("x.npy", allow_pickle=True)
#     y = np.load("y.npy", allow_pickle=True)
#     z = np.load("z.npy", allow_pickle=True)
#     file_name = "pinn"
#     np_to_vtp("pinn_visco_" + str(nu), x, y, z, surf_name='train_loss', log=True, zmax=-1, interp=-1)


nu = 0.01


def normalize_weights(weights, origin):
    return [
        w * np.linalg.norm(wc) / np.linalg.norm(w)
        for w, wc in zip(weights, origin)
    ]


class RandomCoordinates(object):
    def __init__(self, origin):
        self.origin_ = origin
        self.v0_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )
        self.v1_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc
            for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]


class LossSurface(object):
    def __init__(self, model, inputs, outputs, X_f_train_f):
        self.a_grid_ = None
        self.b_grid_ = None
        self.loss_grid_ = None
        self.model_ = model
        self.inputs_ = inputs
        self.outputs_ = outputs
        self.xf = X_f_train_f
        self.no_of_interior_points = 8000
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def residual_loss(self, ):
        with tf.GradientTape() as tape:
            tape.watch(self.xf)
            with tf.GradientTape() as tape2:
                u_predicted = self.model_(self.xf)
            grad = tape2.gradient(u_predicted, self.xf)
            du_dx = grad[:, 0]
            du_dt = grad[:, 1]
        j = tape.gradient(grad, self.xf)
        d2u_dx2 = j[:, 0]

        u_predicted = tf.cast(u_predicted, dtype=tf.float64)
        du_dx = tf.reshape(du_dx, [self.no_of_interior_points + 456, 1])
        d2u_dx2 = tf.reshape(d2u_dx2, [self.no_of_interior_points + 456, 1])
        du_dt = tf.reshape(du_dt, [self.no_of_interior_points + 456, 1])
        f = du_dt + u_predicted * du_dx - (nu / 3.14 * d2u_dx2)
        f = tf.math.reduce_mean(tf.math.square(f))
        return f

    def loss_total(self, ):
        y_pred = self.model_(self.inputs_)
        loss_data = self.loss_fn(y_pred, self.outputs_)
        loss_res = self.residual_loss()
        loss = tf.reduce_mean(tf.square(loss_data)) + tf.reduce_mean(tf.square(loss_res))
        return loss

    def compile(self, range_val, points, coords):
        a_grid = tf.linspace(-1.0, 1.0, num=points) ** 3 * range_val
        b_grid = tf.linspace(-1.0, 1.0, num=points) ** 3 * range_val
        loss_grid = np.empty([len(a_grid), len(b_grid)])
        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                self.model_.set_weights(coords(a, b))
                loss = self.loss_total()
                loss_grid[j, i] = loss
        print(loss_grid)
        self.model_.set_weights(coords.origin_)
        self.a_grid_ = a_grid
        self.b_grid_ = b_grid
        self.loss_grid_ = loss_grid

    def plot(self, range_val=1.0, points=24, levels=70, ax=None, **kwargs):
        xs = self.a_grid_
        ys = self.b_grid_
        zs = self.loss_grid_
        # if ax is None:
        #     #ax = plt.figure().add_subplot(projection='3d')
        #     _, ax = plt.subplots(**kwargs)
        #     ax.set_title("Loss Surface With trajectories")
        #     ax.set_aspect("equal")
        #
        # # Set Levels
        # min_loss = zs.min()
        # max_loss = zs.max()
        # levels = tf.exp(
        #     tf.linspace(
        #         tf.math.log(min_loss), tf.math.log(max_loss), num=levels
        #     )
        # )
        # # Create Contour Plot
        # CS = ax.contour(
        #     xs,
        #     ys,
        #     zs,
        #     levels=levels,
        #     cmap="magma",
        #     linewidths=0.75,
        #     norm=mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
        # )
        #
        # ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        # # surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
        # #                        linewidth=0, antialiased=False)

        return ax, xs, ys, zs


def vectorize_weights_(weights):
    vec = [w.flatten() for w in weights]
    vec = np.hstack(vec)
    return vec


def vectorize_weight_list_(weight_list):
    vec_list = []
    for weights in weight_list:
        vec_list.append(vectorize_weights_(weights))
    weight_matrix = np.column_stack(vec_list)
    return weight_matrix


def shape_weight_matrix_like_(weight_matrix, example):
    weight_vecs = np.hsplit(weight_matrix, weight_matrix.shape[1])
    sizes = [v.size for v in example]
    shapes = [v.shape for v in example]
    weight_list = []
    for net_weights in weight_vecs:
        vs = np.split(net_weights, np.cumsum(sizes))[:-1]
        vs = [v.reshape(s) for v, s in zip(vs, shapes)]
        weight_list.append(vs)
    return weight_list


def get_path_components_(training_path, n_components=2):
    weight_matrix = vectorize_weight_list_(training_path)
    pca = PCA(n_components=2, whiten=True)
    components = pca.fit_transform(weight_matrix)
    example = training_path[0]
    weight_list = shape_weight_matrix_like_(components, example)
    return pca, weight_list


class PCACoordinates(object):
    def __init__(self, training_path):
        self.v1_ = None
        self.v0_ = None
        self.origin_ = None
        origin = training_path[-1]
        self.pca_, self.components = get_path_components_(training_path)
        self.set_origin(origin)

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc
            for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]

    def set_origin(self, origin, renorm=True):
        self.origin_ = origin
        if renorm:
            self.v0_ = normalize_weights(self.components[0], origin)
            self.v1_ = normalize_weights(self.components[1], origin)


def weights_to_coordinates(coords, training_path):
    components = [coords.v0_, coords.v1_]
    comp_matrix = vectorize_weight_list_(components)
    comp_matrix_i = np.linalg.pinv(comp_matrix)
    w_c = vectorize_weights_(training_path[-1])
    coord_path = np.array(
        [
            comp_matrix_i @ (vectorize_weights_(weights) - w_c)
            for weights in training_path
        ]
    )
    return coord_path


def plot_training_path(coords, training_path, ax=None, end=None, **kwargs):
    path = weights_to_coordinates(coords, training_path)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    colors = range(path.shape[0])
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0, end)
    ax.scatter(
        path[:, 0], path[:, 1], s=4, c=colors, cmap="cividis", norm=norm,
    )
    return ax, path, colors

def get_splope_bias_from_points(r1,r2):
    x1,y1=r1
    x2,y2=r2
    m=(y2-y1)/(x2-x1)
    b=y2-m*x2
    return m,b