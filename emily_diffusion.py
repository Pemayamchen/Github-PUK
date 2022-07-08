# %%
from RandomWalkSims import *
import numpy as np

num_tracks = 900 # number of tracks to make
min_parent_len = 5 # min length
dim = 2
Ds = np.random.uniform(low = 10**-3, high = 1, size=(num_tracks)) #np.array([1]*num_tracks) # D
dt = 1 # temporal resolution
sigmaND = [0]*num_tracks # no localization error so 0
NsND = np.array([99]*num_tracks) # number of steps to take from starting point so length you want -1

# setting fixed confinement size instead of track length dependent
Bmin, Bmax = 0.05, 0.5 # confinement radius
B = np.random.uniform(Bmin, Bmax, size=num_tracks) 
angles = np.random.randint(0,180, size=(num_tracks,3))
ellipse_dims = np.random.uniform(Bmin, Bmax, size=(num_tracks,3)) 
r_c = np.sqrt(Ds * NsND * dt / B)  # solving for r_c in eq. 8 Kowalek

Rmin, Rmax = 5, 25 # proportion of active to passive motion
R = np.random.uniform(Rmin, Rmax, size=num_tracks)
TDM = NsND * dt
vs = np.sqrt(R * 4 * Ds / TDM)  # solving for v in eq. 7 Kowalek

subalphas = np.random.uniform(0, 0.7, size=num_tracks)
superalphas = np.random.uniform(1.3, 2, size=num_tracks)

# ND: normal diffusion, DM: directed motion, CD: confined, SD: subdiffusive
ND_diff = Gen_normal_diff(Ds, dt, sigmaND, NsND, dim=dim, min_len=min_parent_len, withlocerr=False)
DM_diff = Gen_directed_diff(Ds, dt, vs, sigmaND, NsND, dim=dim, min_len=min_parent_len, withlocerr=False)
# # if you want s
SD_diff = Gen_anomalous_diff(Ds, dt, subalphas, sigmaND, NsND, dim=dim, min_len=min_parent_len, withlocerr=False)
# #SupD_diff = Gen_anomalous_diff(Ds, dt, superalphas, sigmaND, NsND, dim=dim, min_len=min_parent_len)

CD_diff = Gen_new_confined_diff(Ds, dt, sigmaND, NsND, dim=dim, min_len=min_parent_len, initial_state=[],
                                ellipse_dims=ellipse_dims, angles=angles, withlocerr=False)

# %%

# print(len(ND_diff),len(DM_diff),len(CD_diff),len(SD_diff))

# print(ND_diff[0])
# print(ND_diff[0].shape,DM_diff[0].shape,CD_diff[0].shape,SD_diff[0].shape)

# graphs of the 3 types of motion

# plt.figure()
plt.plot(ND_diff[0][:,0],ND_diff[0][:,1],'m',label="normal diffusion")
# plt.show()

# plt.figure()
plt.plot(SD_diff[0][:,0],SD_diff[0][:,1],'g',label="subdiffusion")
# plt.show()

# plt.figure()
plt.plot(DM_diff[0][:,0],DM_diff[0][:,1],'c',label="directed motion")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('x-direction',fontsize=14)
plt.ylabel('y-direction',fontsize=14)
plt.legend(fontsize=11.5)
# plt.show()


# %%
# changing the ouput format to matrices

# for normal diffusion
ND_diff = np.vstack(ND_diff)
NDx,NDy = ND_diff[:,0],ND_diff[:,1]
# print(np.where(x == 0))
NDx = NDx.reshape(900,-1)
NDy = NDy.reshape(900,-1)
print(NDx.shape,NDy.shape)

# for directed motion
DM_diff = np.vstack(DM_diff)
DMx,DMy = DM_diff[:,0],DM_diff[:,1]
DMx = DMx.reshape(900,-1)
DMy = DMy.reshape(900,-1)
print(DMx.shape,DMy.shape)

# for subdiffusive motion
SD_diff = np.vstack(SD_diff)
SDx,SDy = SD_diff[:,0],SD_diff[:,1]
SDx = SDx.reshape(900,-1)
SDy = SDy.reshape(900,-1)
print(SDx.shape,SDy.shape)

# %%
# for traces format
ND_traces = np.split(ND_diff,900,axis=0)
DM_traces = np.split(DM_diff,900,axis=0)
SD_traces = np.split(SD_diff,900,axis=0)

# %%
# MSD calculations

def SquareDist(x0, x1, y0, y1):
    return (x1 - x0) ** 2 + (y1 - y0) ** 2

def msd(t, frac):
    if t.shape[1]==2:
        x = t[:,0]
        y = t[:,1]
    elif t.shape[1]==3:
        x = t[:,0]
        y = t[:,1]

    N = int(len(x) * frac) if len(x)>10 else len(x)
    msd = []
    for lag in range(1, N):
        msd.append(np.mean([SquareDist(x[j], x[j + lag], y[j], y[j + lag])
        for j in range(len(x) - lag)]))
    return np.array(msd)

MSDs = []
for t in ND_traces[800:]:
    MSDs.append(msd(t, 1))
MSDs = np.array(MSDs, dtype=object)

# %%
# random color generator
import random

def random_color():
    rgbl=[round(random.uniform(0, 1),1),round(random.uniform(0, 1),1),round(random.uniform(0, 1),1)]
    return tuple(rgbl)

# %%
# Linear regression on MSD

from scipy.optimize import curve_fit
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

def msd_func(x,D,alpha):
    return 2*dim*D*(x)**alpha

x_data_list = []
slope_list = []
error_list = []
diff_coeff_list = []  
color_list = []
MSDs = MSDs.astype('float64')
remove_list = []

for j in range(len(MSDs)):
    color_list.append(random_color())
    start_time = 0
    start_index = int(start_time/(1/30))
    end_time = len(MSDs[j])
    end_index = int(end_time/(1/30))
    x_data = [*range(0,len(MSDs[j]))]
    x_data_list.append(np.array(x_data))
    xx = x_data_list[j][:,np.newaxis]
    xx = xx.astype('float64')
    a,resid,_,_ = np.linalg.lstsq(xx, MSDs[j], rcond=None)
    # plt.plot(MSDs[j],color = color_list[j],label='MSD')
    # plt.plot(xx, a*xx, color = color_list[j], linestyle='dashed', label=r'$y=4*D*t$')
    # plt.xlabel(r'$\tau$')
    # plt.legend()
    # plt.figure()
    # r_squared = 1 - (resid / (MSDs[j].size * MSDs[j].var()))
    r_squared = 1 - resid / (sum((MSDs[j] - MSDs[j].mean())**2))
    error_list.append(r_squared)
    msds = MSDs[j]
    x = np.arange(1,len(msds) + 1)*dt
    popt,pcov = curve_fit(msd_func,x,msds,maxfev=800)
    
    diff_coeff = popt[0]

    if diff_coeff > 3:
        remove_list.append(j)
    
    diff_coeff_list.append(diff_coeff)
    
# print(diff_coeff_list)
# print(Ds)
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
            
delete_multiple_element(diff_coeff_list,remove_list)
Dstest = Ds[800:].tolist()
delete_multiple_element(Dstest,remove_list)

plt.hist(np.array(diff_coeff_list),bins=int(np.sqrt(100)))
plt.xlabel('Diffusion coefficients')
plt.ylabel('Frequency')
# plt.xlim(-0.5,4.5)
plt.show()

# %%
def t_msd_func(x,D):
    return 2*dim*D*(x)**1

t_x_data_list = []
t_slope_list = []
t_error_list = []
t_diff_coeff_list = []  
t_color_list = []
t_MSDs = MSDs.astype('float64')
t_remove_list = []

for j in range(len(t_MSDs)):
    t_color_list.append(random_color())
    t_start_time = 0
    t_start_index = int(t_start_time/(1/30))
    t_end_time = len(t_MSDs[j])
    t_end_index = int(t_end_time/(1/30))
    t_x_data = [*range(0,len(t_MSDs[j]))]
    t_x_data_list.append(np.array(t_x_data))
    t_xx = t_x_data_list[j][:,np.newaxis]
    t_xx = t_xx.astype('float64')
    t_a,t_resid,_,_ = np.linalg.lstsq(t_xx, t_MSDs[j], rcond=None)
    # plt.plot(MSDs[j],color = color_list[j],label='MSD')
    # plt.plot(xx, a*xx, color = color_list[j], linestyle='dashed', label=r'$y=4*D*t$')
    # plt.xlabel(r'$\tau$')
    # plt.legend()
    # plt.figure()
    # r_squared = 1 - (resid / (MSDs[j].size * MSDs[j].var()))
    t_r_squared = 1 - t_resid / (sum((t_MSDs[j] - t_MSDs[j].mean())**2))
    t_error_list.append(t_r_squared)
    t_msds = t_MSDs[j]
    t_x = np.arange(1,len(t_msds) + 1)*dt
    popt,pcov = curve_fit(t_msd_func,t_x,t_msds,maxfev=800)
    
    t_diff_coeff = popt[0]

    if t_diff_coeff > 3:
        t_remove_list.append(j)
    
    t_diff_coeff_list.append(t_diff_coeff)
    
# print(diff_coeff_list)
# print(Ds)
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
            
delete_multiple_element(t_diff_coeff_list,t_remove_list)
t_Dstest = Ds[800:].tolist()
delete_multiple_element(t_Dstest,t_remove_list)
# %%
plt.hist(np.array(t_Dstest),bins=10,color='blue',alpha=1,edgecolor='darkblue',label='Original D')
plt.hist(np.array(diff_coeff_list),bins=20,color='magenta',alpha=0.5,edgecolor=(1, 0, 0, 1),label=r'MSD with $\alpha$')
plt.hist(np.array(t_diff_coeff_list),bins=20,color='green',alpha=0.25,edgecolor='darkgreen',label=r'MSD $\alpha=1$')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('D predicted by MSD direct motion',fontsize = 14)
plt.ylabel('Frequency', fontsize=14)
# plt.xlim(-0.5,4.5)
# print(max(diff_coeff_list))
plt.legend(fontsize=13)
plt.show()
# plt.hist(np.array(diff_coeff_list),bins=int(np.sqrt(100)))
# plt.xlabel('Diffusion coefficients')
# plt.ylabel('Frequency')
# # plt.xlim(-0.5,4.5)
# plt.show()

# %%
# print(np.array(error_list[:4]))
plt.hist(np.array(error_list),weights=np.ones_like(error_list) / len(error_list), bins=20, color='magenta',alpha=0.5,edgecolor=(1, 0, 0, 1),label=r'MSD with $\alpha$')
plt.hist(np.array(t_error_list),weights=np.ones_like(t_error_list) / len(t_error_list), bins=20, color='green',alpha=0.25,edgecolor='darkgreen',label=r'MSD $\alpha=1$')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Determination coefficient $R^2$',fontsize=14)
plt.ylabel('Relative frequency',fontsize=14)
plt.legend(fontsize=14)
# plt.xlim(-3.5,1.5)
plt.show()
# %%
# MSE
from sklearn.metrics import mean_squared_error
mean_squared_error(Dstest,diff_coeff_list)
mean_squared_error(Dstest,t_diff_coeff_list)

# %%
plt.scatter(Ds, diff_coeff_list)
plt.xlabel('D in simulations')
plt.ylabel('D predicted by MSD')

# %%
# splitting data into training and testing data
NDx_train = NDx[:800]
NDy_train = NDy[:800]

DMx_train = DMx[:800]
DMy_train = DMy[:800]

SDx_train = SDx[:800]
SDy_train = SDy[:800]

NDx_test = NDx[800:]
NDy_test = NDy[800:]

DMx_test = DMx[800:]
DMy_test = DMy[800:]

SDx_test = SDx[800:]
SDy_test = SDy[800:]

D_train = Ds[:800]
D_test = Ds[800:]

# %%
# GPR training

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e10))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
# %%
gaussian_process.fit(DMx_train,D_train)
gx = gaussian_process.predict(DMx_test)

gaussian_process.fit(DMy_train,D_train)
gy = gaussian_process.predict(DMy_test)

result = np.add(gx,gy)/2

# %%
from sklearn.metrics import mean_squared_error
print(mean_squared_error(D_test,result))

# %%
# MLP neural network training

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

xy_train = np. concatenate((DMx_train,DMy_train),axis = 1)
diffus_train = D_train
xy_test = np. concatenate((DMx_test,DMy_test),axis = 1)
diffus_test = D_test
regr = MLPRegressor(hidden_layer_sizes=(200,20),learning_rate='constant',learning_rate_init=0.01, random_state=1, max_iter=1000).fit(xy_train, diffus_train)
rx = regr.predict(xy_test)

print(diffus_test.shape)
print(xy_train.shape,diffus_train.shape)
# print(d_train)

mean_squared_error(diffus_test,rx)

# %%
plt.hist(Ds[800:],bins=10,color='blue',alpha=1,edgecolor='darkblue',weights=np.ones_like(Ds[800:]) / len(Ds[800:]),label='Original D')
plt.hist(np.array(diff_coeff_list),bins=20,color='orange', alpha=0.5,edgecolor=(1, 0, 0, 1),weights=np.ones_like(diff_coeff_list) / len(diff_coeff_list),label=r'MSD test')
plt.hist(np.array(result),bins=10,color='red',alpha=0.25,edgecolor=(1, 0, 0, 1),weights=np.ones_like(result) / len(result),label=r'GPR test')
plt.hist(rx,color='cyan',bins=10, alpha=0.50,edgecolor=(1, 0, 0, 1),weights=np.ones_like(rx) / len(rx),label=r'NN test')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Diffusion coefficients directed motion',fontsize = 15)
plt.ylabel('Relative frequency', fontsize=15)
# plt.xlim(-0.5,4.5)
# print(max(diff_coeff_list))
plt.legend(fontsize=13)
plt.show()