import theano
import starry
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import exoplanet as xo
import scipy
from scipy import constants
import os, sys
import csv
import pandas as pd
import time as timer
from astropy import constants as const

starry.config.quiet = True

def Planck(wavelength,T):
    wavelength=wavelength
    S = (2*scipy.constants.c**2*scipy.constants.h/wavelength**5)/(
        np.exp(scipy.constants.h*scipy.constants.c/(wavelength*scipy.constants.k*T))-1)
    return S
def brightnesstemp(wavelength,j):
    tbright=(scipy.constants.h*scipy.constants.c/
             (wavelength*scipy.constants.k*np.log(2*scipy.constants.h*scipy.constants.c**2/
                                                  (j*wavelength**5)+1)))
    return tbright

def binning_data(data, size):
    '''
    Median bin an array.

    Parameters
    ----------
    data     : 1D array
        Array of data to be binned.

    size     : int
        Size of bins.

    Returns
    -------
    binned_data: 1D array
        Array of binned data.

    binned_data: 1D array
        Array of standard deviation for each entry in binned_data.
    '''
    #data = np.ma.masked_invalid(data) 
    reshaped_data   = data.reshape((int(len(data)/size), size))
    binned_data     = np.median(reshaped_data[1::], axis=1)
    binned_data_std = np.std(reshaped_data, axis=1)
    return binned_data, binned_data_std

#List of AOR numbers
AOR_list = ['46471424','46471168','46470912','46470656','46470400','46469632','46469120','46468608','46468352','46468096']

#Loop through the AORs, bin the data, and assign an observation number to each.
timeAll=[]
fluxAll=[]
xcenAll=[]
ycenAll=[]
EDNumAll=[]
rangeAll = []
slopeAll = []
for idx,AOR in enumerate(AOR_list):
    #Read data for AOR
    df =pd.read_csv('/Users/dylanskeating/Documents/Research/HARMONiE/XO-3b/Photometry/Data/XO3_r'
                    +AOR+'_phot_ch2.txt')
    #Bin the data
    time, _ = binning_data(np.asarray(df['BMJD']), 64)
    time -=time[0]
    flux, _ = binning_data(np.asarray(df['flux_2p75']-df['bkgd']), 64)
    flux /=np.median(flux)
    fluxrange =np.std(flux)
    slope = (flux[-1]-flux[0])/(time[-1]-time[0])
    xcentroid,_ = binning_data(np.asarray(df['xcen']), 64)
    xcentroid-=15
    ycentroid,_ = binning_data(np.asarray(df['ycen']), 64)
    ycentroid-=15
    EDNum = np.ones_like(time)*(idx)
    timeAll=np.append(timeAll,time)
    fluxAll=np.append(fluxAll,flux)
    rangeAll=np.append(rangeAll,fluxrange*np.ones_like(time))
    slopeAll=np.append(slopeAll,slope*np.ones_like(time))
    xcenAll=np.append(xcenAll,xcentroid)
    ycenAll=np.append(ycenAll,ycentroid)
    EDNumAll = np.append(EDNumAll,EDNum)
#Make the big ass dataframe
data = {'ED Num':EDNumAll,'time':timeAll,'flux':fluxAll,'xcen':xcenAll,'ycen':ycenAll,'fluxrange':rangeAll,'slopeall':slopeAll}

EDdf = pd.DataFrame(data)
EDdf['ED Num']=EDdf['ED Num'].astype(int)


criteria1=EDdf['ED Num'] == 0
criteria2=EDdf['ED Num'] == 1

EDdf1=EDdf.loc[criteria1 | criteria2]
time_XO = EDdf1['time'][EDdf1['ED Num'] ==1]
flux_XO = EDdf1['flux'][EDdf1['ED Num'] ==1]

#Make the Starry Mean function
def Eclipse(P,a,ED,rp_rstar,inc,t0,time,ecc,omega):
    A = starry.Primary(
        starry.Map(ydeg=0, udeg=0, amp=1.0), m=1.0, r=1.0
    )


    # Instantiate the planet. Everything is fixed except for
    # its luminosity and the hot spot offset.


    b = starry.Secondary(
        starry.Map(ydeg=0, udeg=0,amp=0,inc=inc, obl=0.0),
        m=0.0,
        r=rp_rstar,
        prot=P,
        porb=P,
        t0=t0,
        a =a,
        ecc=ecc,
        w=omega
    )
    sys=starry.System(A, b)
    return 1+ED*(rp_rstar**2-1+sys.flux(time))/rp_rstar**2


P_XO = 3.19153285
a_XO = 7.052
inc_XO = 84.20
radius_XO = (1.217*const.R_jup/(1.4517572*const.R_sun)).value
EDmin=(0.25)**0.25*(1-0.8)**0.25*np.sqrt(1./a_XO)*radius_XO**2
EDmax=(2./3.)**0.25*np.sqrt(1./a_XO)*radius_XO**2

import theano.tensor as tt

with pm.Model() as model:
    #Roughly estimate upper limit for eclipse depth--- limit of no heat recirculation, Ab=1
    EDmax=(2./3.)**0.25*np.sqrt(1./a_XO)*radius_XO**2

    ED_XO_mean_offset = pm.Normal('ED XO mean offset',0,1)
    ED_XO_mean = pm.Deterministic('ED XO mean',EDmax/2+ED_XO_mean_offset*EDmax/2)
    #ED_XO_sig_offset = pm.HalfNormal('ED XO sig offset',1)
    ED_XO_sig = pm.HalfNormal('ED XO sig',300*10**-6)
    #ED_XO_offset = pm.Normal('ED XO offset',0,1,shape=10)

    #ED_XO = pm.Deterministic('ED XO',ED_XO_mean*tt.ones(10)+ED_XO_offset*ED_XO_sig)
    ED_XO_offset = pm.Normal('ED XO offset',0,1,shape=10)
    ED_XO = pm.Deterministic('ED XO',ED_XO_mean+ED_XO_offset*ED_XO_sig)

    radius_XO = (1.217*const.R_jup/(1.4517572*const.R_sun)).value
    P_XO = 3.19153285
    a_XO = 7.052
    inc_XO = 84.20
    ecc_XO = 0.2769#pm.Normal('ecc',0.27587,0.00069)
    omega_XO = 347.2#pm.Normal('w',349.35,0.675)
    
    
    t0_XO_offset = pm.Normal('t0 XO offset',mu=0,sd=1,shape=10)
    t0_XO = pm.Deterministic('t0 XO ch2',(np.max(EDdf['time'].values)+np.min(EDdf['time'].values))/2+
                             t0_XO_offset*(np.max(EDdf['time'].values)+np.min(EDdf['time'].values))/2)
    
    #compute eclipse signal for each observation
    Eclipse_0 = pm.Deterministic('Eclipse 0',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[0],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[0],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==0].values))
    Eclipse_1 = pm.Deterministic('Eclipse 1',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[1],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[1],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==1].values))
    Eclipse_2 = pm.Deterministic('Eclipse 2',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[2],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[2],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==2].values))
    Eclipse_3 = pm.Deterministic('Eclipse 3',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[3],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[3],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==3].values))
    Eclipse_4 = pm.Deterministic('Eclipse 4',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[4],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[4],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==4].values))
    Eclipse_5 = pm.Deterministic('Eclipse 5',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[5],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[5],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==5].values))
    Eclipse_6 = pm.Deterministic('Eclipse 6',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[6],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[6],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==6].values))
    Eclipse_7 = pm.Deterministic('Eclipse 7',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[7],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[7],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==7].values))
    Eclipse_8 = pm.Deterministic('Eclipse 8',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[8],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[8],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==8].values))
    Eclipse_9 = pm.Deterministic('Eclipse 9',Eclipse(P=P_XO,a=a_XO,ED=ED_XO[9],rp_rstar=radius_XO,
                                                                 inc=inc_XO,t0=t0_XO[9],ecc=ecc_XO,omega=omega_XO,
                                                                 time=EDdf['time'].loc[EDdf['ED Num']==9].values))
    #Gaussian process params for Channel 2
    lx_XO = pm.InverseGamma('lx XO',alpha=11,beta=5)
    ly_XO = pm.InverseGamma('ly XO',alpha=11,beta=5)
    eta_XO = pm.HalfNormal('eta XO',(np.mean(EDdf['fluxrange'].unique()))/3)    
    #Fit for the error
    eps_XO = pm.HalfNormal('sig XO',(EDdf['fluxrange'].unique())/3,shape=10)
    
    cov_func_XO = eta_XO*pm.gp.cov.ExpQuad(input_dim=3, ls=[lx_XO,ly_XO], active_dims=[1, 2])

    
    gp_XO = pm.gp.Marginal(cov_func = cov_func_XO)
    #Residuals
    resid_0= EDdf['flux'].loc[EDdf['ED Num']==0].values - Eclipse_0 #- (astro_m_XO[0]*EDdf['time'].loc[EDdf['ED Num']==0].values+astro_b_XO[0])
    resid_1= EDdf['flux'].loc[EDdf['ED Num']==1].values - Eclipse_1 #- (astro_m_XO[1]*EDdf['time'].loc[EDdf['ED Num']==1].values+astro_b_XO[1])
    resid_2= EDdf['flux'].loc[EDdf['ED Num']==2].values - Eclipse_2 #- (astro_m_XO[2]*EDdf['time'].loc[EDdf['ED Num']==2].values+astro_b_XO[2])
    resid_3= EDdf['flux'].loc[EDdf['ED Num']==3].values - Eclipse_3 #- (astro_m_XO[3]*EDdf['time'].loc[EDdf['ED Num']==3].values+astro_b_XO[3])
    resid_4= EDdf['flux'].loc[EDdf['ED Num']==4].values - Eclipse_4 #- (astro_m_XO[4]*EDdf['time'].loc[EDdf['ED Num']==4].values+astro_b_XO[4])
    resid_5= EDdf['flux'].loc[EDdf['ED Num']==5].values - Eclipse_5 #- (astro_m_XO[5]*EDdf['time'].loc[EDdf['ED Num']==5].values+astro_b_XO[5])
    resid_6= EDdf['flux'].loc[EDdf['ED Num']==6].values - Eclipse_6 #- (astro_m_XO[6]*EDdf['time'].loc[EDdf['ED Num']==6].values+astro_b_XO[6])
    resid_7= EDdf['flux'].loc[EDdf['ED Num']==7].values - Eclipse_7 #- (astro_m_XO[7]*EDdf['time'].loc[EDdf['ED Num']==7].values+astro_b_XO[7])
    resid_8= EDdf['flux'].loc[EDdf['ED Num']==8].values - Eclipse_8 #- (astro_m_XO[8]*EDdf['time'].loc[EDdf['ED Num']==8].values+astro_b_XO[8])
    resid_9= EDdf['flux'].loc[EDdf['ED Num']==9].values - Eclipse_9 #- (astro_m_XO[9]*EDdf['time'].loc[EDdf['ED Num']==9].values+astro_b_XO[9])
    #Model with GP
    yXO_0 = gp_XO.marginal_likelihood("f XO 0", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==0].values,y=resid_0, noise=eps_XO[0])
    yXO_1 = gp_XO.marginal_likelihood("f XO 1", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==1].values,y=resid_1, noise=eps_XO[1])
    yXO_2 = gp_XO.marginal_likelihood("f XO 2", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==2].values,y=resid_2, noise=eps_XO[2])
    yXO_3 = gp_XO.marginal_likelihood("f XO 3", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==3].values,y=resid_3, noise=eps_XO[3])
    yXO_4 = gp_XO.marginal_likelihood("f XO 4", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==4].values,y=resid_4, noise=eps_XO[4])
    yXO_5 = gp_XO.marginal_likelihood("f XO 5", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==5].values,y=resid_5, noise=eps_XO[5])
    yXO_6 = gp_XO.marginal_likelihood("f XO 6", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==6].values,y=resid_6, noise=eps_XO[6])
    yXO_7 = gp_XO.marginal_likelihood("f XO 7", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==7].values,y=resid_7, noise=eps_XO[7])
    yXO_8 = gp_XO.marginal_likelihood("f XO 8", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==8].values,y=resid_8, noise=eps_XO[8])
    yXO_9 = gp_XO.marginal_likelihood("f XO 9", X=EDdf[['time','xcen','ycen']].loc[EDdf['ED Num'] ==9].values,y=resid_9, noise=eps_XO[9])
        

tic =timer.time()
with model:
    trace = pm.sample(draws=1000, tune=2000,chains=4,cores=1,step=xo.get_dense_nuts_step(target_accept=0.9))
   # trace = pm.sample(draws=1000, tune=2000,chains=4,cores=1,target_accept=0.9)
tok = timer.time()
print('time taken (minutes):',(tok-tic)/60)
pm.save_trace(trace,'/Users/dylanskeating/Documents/Research/HARMONiE/XO-3b/Results/XO_3b_PooledGP_HierED_2',overwrite=True)
var_names=['ED XO mean','ED XO sig','ED XO','lx XO','ly XO','eta XO','t0 XO ch2','sig XO']
print(pm.summary(trace,var_names=var_names,round_to=7))

with model:
    pm.traceplot(trace,var_names=var_names)
    plt.savefig('/Users/dylanskeating/Documents/Research/HARMONiE/XO-3b/Results/XO_3b_PooledGP_HierED_2.pdf',bbox_inches='tight')


