#!/usr/bin/env python
# coding: utf-8

"""Particle Filter of biomass respiration profiles in dynamically operated bioreactors"""
STRICT = False
SEED = 5

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed, uniform, randn
import scipy.stats
#from scipy.stats import truncnorm
from multiprocessing import Pool, cpu_count
from functools import partial

import time
import pandas
import pickle

#https://github.com/ipython/ipython/issues/11098)
import plotting # needs to be in a separate cell Jupyter Notebooks
import environment
from getdata import *

EXPORT = False
#EXPORT = True

################################################################################
################## FUNCTIONS FOR PARTICLE FILTER RECONSTRUCTION ################
################################################################################

def export(df, fname=None):
    if not EXPORT: return
    if type(df) != pandas.core.frame.DataFrame:
        df = pandas.DataFrame(df)
    fname = fname or 'temp'
    fname = 'export/data/{}-{}.csv'.format(fname, time.strftime('%Y%m%d-%H%M'))
    df.to_csv(fname)

def save_fig(fpathname):
    if not EXPORT: return
    print("Saving figure: {}".format(fpathname))
    plt.savefig(fpathname+".png", dpi=300)

def save_data(data, UR_mv, fname):
    if not EXPORT: return
    sh = UR_mv.shape
    output = np.zeros((sh[0], sh[1]+3))
    output[:,:-3] = UR_mv
    output[:,-3] = -UR_mv[:,2]/UR_mv[:,1] #-CUR/OUR
    output[:,-2] = np.cumsum(UR_mv[:,1], axis=0) #O2
    output[:,-1] = -np.cumsum(UR_mv[:,2], axis=0) #CO2
    df = pandas.DataFrame(output, index=data[:,0], columns=['NUR', 'OUR', 'CUR', 'RQ', 'O2SUM', 'CO2SUM'])
    export(df, fname)

def create_uniform_particles(URlow, URhigh, initialstate, N):
    #initial state: N2l, O2l, CO2l, N2b, O2b, CO2b, N2h, O2h, CO2h, CO2hydr
    v, w = len(initialstate), len(URlow)
    particles = np.empty((N, v+w))

    particles[:, :w] = uniform(URlow, URhigh, size=(N, w))
    particles[:, w:] = initialstate
    return particles

def pre_predict(URs, cov, dUR, steps=1):
    """Change uptake rate UR with noisy dUR, taking into account the covariance

    the covariance matrix must be square [wxw] with w the number of gas compounds
    dUR can be either a vector of length w or a scalar and is in unit mol/s
    """
    N, w = len(URs), len(cov)
    mean = np.zeros(w)
    URcov = np.multiply(cov, steps*dUR**2)
    URs += np.random.multivariate_normal(mean, URcov, size=N)

    #This solution allows for the OUR and CER to achieve inversed sign values which does not represent true biological activity.
    #More accurate solutions would attempt to include uptake rate boundaries:

    # A) lower bound fix to 0 (50% of particles useless, when UR close to 0)
    # B) flip sign for <0 (more difficult to achieve UR==0)
    # C) implement one-sided truncation scipy.stats.truncnorm(-S*dOUR, 1E6, size=N) (SLOW due to performance bug in scipy)

    # Implementation of A) & B) 
    #Additional constraints for UptakeRates (UR) as behavior close to 0 is stranuous
    #p[p[:,0]<0,0] = 0 #oxygen cannot be produced, co2 cannot be consumed
    #p[p[:,1]>0,1] = 0
    #p[p[:,0]<0,0] *= -1 #
    #p[p[:,1]>0,1] *= -1 
    #p[p[:,0]<0,0] *= -0.1 #Idem to above but with close bind to 0
    #p[p[:,1]>0,1] *= -0.1 

    # Implementation of C) Truncated normal distribution solution
    # #Truncnorm code from 1.3.3 to a custom module is >100x times faster than 1.5.x
    # #numpy runs into issues when the bounded range is close to 37.5 standard diviations
    # #truncate all ranges to max 20 standard divs
    # stddOUR = -p[:,0]/dOUR
    # stddOUR[stddOUR>20] = 20
    # p[:,0] += scipy.stats.truncnorm.rvs(-20, stddOUR, loc=0) * dOUR
    # stddCER = -p[:,1]/dCER
    # stddCER[stddCER<-20] = -20
    # p[:,1] += scipy.stats.truncnorm.rvs(stddCER, 20, loc=0) * dCER

def predict(t, e, URs, c_l, x_b, x_h, pH, cCO2=None, cCO2hydr=None):
    """Proliferate system state according to the physicochemical model."""
    # Uses global defined process constants:
    # compounds - activity coefficients, hydration radius, etc of charged compounds
    # gasses - ingas and recycle gasses
    # Vb, Vl, Vh, Vm - bubble, liquid, headspace and molar volumes
    # kK - equilibrium constants of CO2 speciation

    TRs = e['Vl'] * e['kLa'] * (e['P'] * e['H'] * x_b - c_l) #Bulk - gas [mol/sec]

    #Gasflow from liquid to headspace
    Fnb2hs = (e['ingas']['Fwet'] + 
            e['recgas']['Fwet'] - 
            TRs.sum(axis=1)/e['Pf'])[:,None] #[mol/sec]

    #Changes in liquid, bulk gas, and headspace gas composition
    dc_ls = (TRs - URs)/e['Vl']    # [mol/sec]
    dx_bs = ((e['ingas']['Fwet'] * e['ingas']['xwet'] + 
        e['recgas']['Fwet'] * x_h) - 
        x_b * Fnb2hs - TRs) / (e['Vb']/e['Vm'])
    dx_hs = Fnb2hs * (x_b - x_h) / (e['Vh']/e['Vm'])

    if cCO2hydr is not None: #Perform CO2 speciation when present
        co2speciation(e, cCO2, cCO2hydr, pH)

    c_l += dc_ls
    x_b += dx_bs
    x_h += dx_hs

    #QC - relatively expensive
    if STRICT:
        LOWLIQC = -1E-3 #[1 mmol error band]
        x_b[np.any(x_b>1, axis=1)] = np.nan         #Gasses cannot be over 100%
        x_b[np.any(x_b<0, axis=1)] = np.nan         #And also not really under 0% (margin)
        x_h[np.any(x_h>1, axis=1)] = np.nan         #Gasses cannot be over 100%
        x_h[np.any(x_h<0, axis=1)] = np.nan         #And also not really under 0% (margin)
        c_l[np.any(c_l<LOWLIQC, axis=1)] = np.nan   #Dissolved gasses cannot be under 0%

def co2speciation(e, cCO2, cCO2hydr, pH):
    """Calculate speciation of carbonates depending on system pH."""
    # CO2 speciation input: environment e, cCO2hydr liquid, pH
    kK = e['co2constants']
    aH      = 10**-pH                
    aOH     = e['Kw']/aH                  
    cH      = aH / e['ions']['H'].gamma       # [H+]  
    #cOH     = aOH / compounds['OH']/gamma     # [OH-] 

    #Activity correction for equilibrium constants
    Ka1 = kK['Ka1']/(e['ions']['H'].gamma * e['ions']['HCO3'].gamma)
    Ka2 = kK['Ka2'] * e['ions']['HCO3'].gamma/(e['ions']['H'].gamma * e['ions']['CO3'].gamma)
    fH2CO3 = 1/(1+Ka1/cH+Ka1*Ka2/cH**2)  # Equilibrium [H2CO3]
    fHCO3  = 1/(1+cH/Ka1+Ka2/cH)         # Equilibrium [HCO3]
    fCO3   = 1/(1+cH/Ka2+cH**2/Ka1)      # Equilibrium [CO3]

    #NOTE: sum of CO2 speciation fractions should be 1 (or mathematically close < 1E-6)
    STRICT and np.testing.assert_allclose(fH2CO3+fHCO3+fCO3, 1, rtol=1E-6)

    dcCO2hydrs = cCO2 * (kK['k1'] * e['cH2O'] + kK['k2'] * aOH) - cCO2hydr * (kK['k_1'] * fH2CO3 + kK['k_2'] * fHCO3 * e['ions']['HCO3'].gamma)
    cCO2hydr += dcCO2hydrs
    cCO2 -= dcCO2hydrs


def update(pred, meas, weights, R):
    """Particle Filter Update step calculates new weights of particles."""
    logpdf = scipy.stats.norm(pred - meas, R).logpdf(0).sum(axis=1) #How likely is the observed measurement given the current prediction
    logpdf -= np.nanmax(logpdf) #We shift the likelihood up to prevent calculation issues for very small numbers (log math)

    #Prevent exceptionally unlikely states to yield zero division errors
    mask = logpdf>-600 
    logpdf[~mask] = -600 #Filter out -inf etc.
    pdf = np.exp(logpdf) 
    #pdf /= sum(pdf) #Idem to NOTE above

    weights *= pdf
    weights += 1.0E-300
    weights /= sum(weights)


def estimate(data, weights):
    """Calculate weighted average of viable system states"""
    mean = np.average(data, weights=weights, axis=0)
    var  = np.average((data - mean)**2, weights=weights, axis=0)
    return mean, var

def neff(weights):
    """Quick method to estimate number of particles that still contributes to the system state."""
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    """Resampling step overwrites unused particles with copies of likely particles"""
    particles[:] = particles[indexes]
    #weights.resize(len(particles))
    #weights.fill(1.0 / len(weights)) 
    #NOTE: Is a hard reset of weights prefered over a maintained likelyness?
    weights[:] = weights[:] + 1/len(particles)
    weights /= sum(weights)

def runPF(N, env, UR0low, UR0high, cov, dUR, gasdata, pHdata, R, initialstate, TITLE, fitplotfunc=None):
    e = env() #Get initialization environment

    #Initial conditions - assume steady state
    #c_l = np.array([0.3404, 0.2334, 0.0123])/1000
    if initialstate is None:
        x_b = e['ingas']['xwet']
        x_h = e['ingas']['xwet']
        c_l = e['P'] * e['H'] * x_b
        cCO2hydr = [0.0809/1000] #TODO: Calc this 
        Y0 = np.concatenate([c_l, x_b, x_h, cCO2hydr])
    else:
        Y0 = initialstate

    #Create N particles, and uniform initial Uptake Rates
    particles = create_uniform_particles(UR0low, UR0high, Y0, N)
    weights = np.ones(N)/N

    # Create named array views
    w = len(cov)
    URs = particles[:, :w]
    c_l = particles[:,w:2*w]
    x_b = particles[:,2*w:3*w]
    x_h = particles[:,3*w:4*w]
    # Specific views for CO2 speciation
    cCO2 = c_l[:,-1]
    cCO2hydr = particles[:,-1]
    # Specfic views for the Oxygen and Carbondioxide UR
    OUR = URs[:, 1]
    CUR = URs[:, 2]

    #Store estimates from particles for each measurement 
    d, v = len(gasdata), len(particles[0])
    est_mean = np.zeros((d,v))
    est_var = np.zeros((d,v))

    #Adjust measurement accuracy R on the fly
    data_ts, data_meas = gasdata[:, 0], gasdata[:, 1:]

    mis = np.diff(data_ts)
    mis = np.append(mis, mis[-1])

    adm = np.abs(np.diff(data_meas, axis=0))
    gadm = np.sum(adm[:,1:], axis=1)
    #gadm = adm[:,2]
    gadm = np.append(gadm, gadm[-2])**2
    gadmc = np.convolve(gadm/mis, [0.1, 0.50, 0.30, 0.10, 0.0], 'same') 

    #gadmc -= gadmc.min()
    gadmc /= gadmc.max() #Between 0 and 1

    #Resample threshold
    #resample_th = 0.5**np.log10(N/100)
    resample_th = N/5
    #resample_th = N #Always resample after measurement

    #Do a full particle filter run

    #prepare plot
    if fitplotfunc:
        fitplotfunc = fitplotfunc(TITLE, data_ts, data_meas)
        fitplotfunc.send(None) #This plots the raw data

    s = 100.0
    r = 100.0 
    #gadmc[:] = 1
    #s = 0
    #r = 1

    try:
        t = data_ts[0]
        for i, data_t in enumerate(data_ts):
            while t <= data_t: #In between measurements
                #print((1/r+(1-1/r)*gadmc[i]), (1+s*gadmc[i]))
                pre_predict(URs, cov, dUR*(1/r+(1-1/r)*gadmc[i]))
                #pre_predict(URs, cov, dUR)
                #pH = pHdata.iloc[pHdata.index.get_loc(t, method='nearest')]
                pH = pHdata.get(t, pHdata[-1])
                predict(t, env(t), URs, c_l, x_b, x_h, pH, cCO2, cCO2hydr)
                t+=1

            update(x_h, data_meas[i]*e['Pf'], weights, R*(1+s*gadmc[i]))
            est_mean[i], est_var[i] = estimate(particles, weights)
            if neff(weights) < resample_th: #Resample 
                indexes = systematic_resample(weights)
                resample_from_index(particles, weights, indexes)

            #pre_predict(URs, cov, dUR*disc[i], mis[i])
            #pre_predict(URs, cov, dUR*(1/r+(1-1/r)*gadmc[i]), mis[i])

            if fitplotfunc:
                meani, vari = np.copy(est_mean[i]), np.copy(est_var[i])
                meani[3*w:4*w] /= e['Pf'] #Represent the head space fraction as dry gas
                vari[3*w:4*w] /= e['Pf']
                fitplotfunc.send((data_t, meani, vari))

    except Exception as e:
        import traceback
        traceback.print_exc()

    return est_mean, est_var

def systematic_resample(weights):
    """ Performs the systemic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    #NOTE: original code from filterpy 
    #https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

################################################################################
################## FUNCTIONS FOR SETTING UP THE PARTICLE FILTER ################
################################################################################

def get_args(N, data, pHdata, env, itern=4):
    #Gas properties:
    #                    N2    O2     CO2
    UR0low  = np.array([0, 0, 0])
    UR0high = np.array([0, 0, 0])
    dUR = env.get('dUR')
    R       = np.array([400,    60,    20])/1E6 #Measurement stddev, smooth results
    cov     = np.diag( [0.,    1.,     1.  ])   #Universal: assume no correllations

    #A single run with optimized parameters
    TITLE = "Particle Filter (N={})".format(N)
    for _ in range(int(itern)):
        yield [N, env, UR0low, UR0high, cov, dUR, data, pHdata, R,  None, TITLE, plotting.plot_fit]
        np.random.random(N) 

def low_high_ppm(N, data, ppm, softner=20, env=None):
    """Calculate the lower and higher bound solution by allowing an error of %ppm in CO2-O2"""
    #Gas properties:
    #                    N2    O2     CO2
    UR0low  = np.array([0,    0E-6, -1E-6])
    UR0high = np.array([0,    1E-6,  0E-6])
    dUR     = np.array([0,    8E-8,  8E-8]) #Normally use 8E-8

    #Measurement error on the MS 
    #To allow a significant fraction of particles to remain, we aim for a cubic root of N 
    R       = np.array([200, 30, 15])*1e-6*softner #measurement stddev, smooth results
    cov     = np.diag( [1.,    1.,     1.  ])   #Universal: assume no correllations

    offsets = np.array([[0,     0,   0,    0],
        [0,  ppm, -ppm,    0],
        [0, -ppm,  ppm,    0],
        [0,  ppm,    0, -ppm],
        [0, -ppm,    0,  ppm]])

    if env is None:
        env = environment.test_environment()

    for offset in offsets:
        TITLE = "PF (N={}) O2 {:+} ppm / CO2 {:+} ppm)".format(N, offset[2], offset[3])
        shifteddata = data+offset*1E-6
        sdd = shifteddata[:, 1:]
        sdd[sdd<0] = 0 #Make sure that no measurement are negative
        yield [N, env, UR0low, UR0high, cov, dUR, shifteddata, R,  None, TITLE, plotting.plot_fit]
        #Ensures that parallel fits start at a new randomization point
        np.random.random(N) 

def average(N, data, pHdata, env, parallel=8):
    # If we can unload one core during parallel computing 
    # without sacrificing calculation time do that
    ncpu = cpu_count()
    poolsize = ncpu - (np.ceil(parallel/(ncpu-1)) == np.ceil(parallel/ncpu))

    with Pool(poolsize) as p:
        ans = np.stack(p.starmap(runPF, get_args(N, data, pHdata, env, parallel)))

    meas = ans[:,0]
    mean = np.median(meas, axis=0)

    Pf = env.get('Pf')
    mean[:,9:12]*=(100/Pf) #Make dry air and convert fraction to percentage
    stddev = np.std(meas, axis=0)

    dof = parallel - 1
    conf95 = scipy.stats.t.ppf(0.975, dof)/np.sqrt(dof) #double sided

    hv = mean + stddev * conf95
    lv = mean - stddev * conf95
    xh_lv = lv[:,9:12]
    xh_hv = hv[:,9:12]
    UR_lv = lv[:,0:3]
    UR_hv = hv[:,0:3]
    UR_mv = mean[:,0:3]

    #tmax, tmin = getFeastFromO2gas(data[:,0], data[:,2], 0.015/100)
    x,y = getFeastEndFromCO2gas(data[:,0], data[:,3], 0.02/100)
    FL = x/60/60 #hours

    TITLE = "PF (NxP={}x{})".format(N, parallel)

    plot = plotting.plot_final(data[:,0]/3600, data[:,1:4]*100, xh_lv, xh_hv, UR_mv, UR_lv, UR_hv, FL=FL)

    fname = 'UR-N{}-{}x'.format(N, parallel)

    if EXPORT:
        print("Storing Pickle data")
        pickledata = [data[:,0]/3600, data[:,1:4]*100, xh_lv, xh_hv, UR_mv, UR_lv, UR_hv, FL]
        pickle.dump(pickledata, open('dataR{}.pickle'.format(R), 'wb'))
        save_data(data, UR_mv, fname)

    return UR_mv

def low_high(env=None):
    N = int(1E4) #number of particles to generate
    ppm = 1      #measurement offset eror of mass spec
    soft = int(np.sqrt(N)/2) #smoothing factor (arbitrary unit)
    soft = 10

    data = x_h_meas

    if env is None:
        env = environment.test_environment()


    with Pool(5) as p:
        ans = np.stack(p.starmap(runPF, low_high_ppm(N, data, ppm, soft, env)))

    Pf = env.get('Pf')
    meas = ans[:,0]
    meas[:,:,9:12]/=Pf #Make dry air
    stddev = np.sqrt(ans[:,1])
    stddev[:,:,9:12]/=Pf #Make stddev also for dry air
    hv = meas + stddev
    lv = meas - stddev
    xh_lv = np.min(lv[:,:,9:12], axis=0)
    xh_hv = np.max(hv[:,:,9:12], axis=0)
    UR_lv = np.min(lv[:,:,:3], axis=0)
    UR_hv = np.max(hv[:,:,:3], axis=0)
    UR_mv = np.average(meas[:,:,:3], axis=0)

    TITLE = "PF (N={}) (e={}ppm) (s={})".format(N, ppm, soft)
    temp = plotting.plot_fit(TITLE, data[:,0]/3600, data[:,1:4], xh_lv, xh_hv, UR_mv, UR_lv, UR_hv)
    temp.send(None)
    sh = UR_mv.shape

    output = np.zeros((sh[0], sh[1]+3))
    output[:,:-3] = UR_mv
    output[:,-3] = -UR_mv[:,2]/UR_mv[:,1] #-CUR/OUR
    output[:,-2] = np.cumsum(UR_mv[:,1], axis=0) #O2
    output[:,-1] = -np.cumsum(UR_mv[:,2], axis=0) #CO2

    df = pandas.DataFrame(output, index=data[:,0], columns=['NUR', 'OUR', 'CUR', 'RQ', 'O2SUM', 'CO2SUM'])

    fname = 'UR-N{}-p{}'.format(N, ppm)
    export(df, fname)
    plt.show(block=False)

################################################################################
############### FUNCTIONS TO RUN A PARTICLE FILTER FROM TEST DATA ##############
################################################################################

def get_pandas_data(cyclename):
    #dataset from: Stouten, Gerben (2018): ExploDiv - Temperature (200 cycles, 8 reactors). 
    #4TU.ResearchData. Dataset. https://doi.org/10.4121/uuid:499ec85a-6d8c-4c02-85c1-f932df4452ef 
    dirname = cyclename[:2]
    fname = '~/Downloads/12693359/cycledata/{}/{}.h5.db'.format(dirname, cyclename)
    with pandas.HDFStore(fname, 'r', complib='blosc') as store:
        d = store.get('data')

    #This old data storage format contains interpolated off-gas data.
    #We choose an arbitrary interval of 3 minutes to create discontinuous data
    CO2data = d['CO2'][::180]
    O2data = d['O2'][::180]
    pHdata = d['pH']
    index = d.index[::180].astype(int)/1E9
    return index, O2data, CO2data, pHdata

if __name__ == "__main__":
    #Example runs for data from Temperature Experiment

    seed(SEED)
    STRICT and np.seterr(all='raise')

    #Example of R7 evening dataset
    R = 7
    hours = 23 #Change to 11 for daylight runs
    minutes = (10)*(R-1)
    if minutes >= 60:
        minutes -= 60
        hours = (hours + 1) % 24

    #Setup to old reactor setup
    T = [40, 40, 30, 30, 20, 20, 25, 35][R-1]
    F_in = [300, 300, 300, 300, 300, 300, 300, 300][R-1] #ml/min #R6
    kLa = [1.78, 1.78, 1.78, 1.78, 1.20, 1.78, 1.78, 2.50][R-1]
    pH_offset = [-0.1, 0.25, 0.25, 0.25, 0.15, 0.1, 0.1, 0.35][R-1] 
    I = [0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075][R-1]
    dUR = [30, 30, 12, 12, 17, 10, 10, 17][R-1] * 1E-8

    #Datasets that show clear different behavior of microbial community

    #Paper plots R5 & R8
    #fname = "R{} 2015-09-04 {:02d}h{:02d}".format(R, hours, minutes) #R5 proper PAPER
    #fname = "R{} 2015-09-17 {:02d}h{:02d}".format(R, hours, minutes) #R8 proper PAPER

    #R6 nice hoard/grow
    #fname = "R{} 2015-09-02 {:02d}h{:02d}".format(R, hours, minutes)
    #fname = "R{} 2015-09-04 {:02d}h{:02d}".format(R, hours, minutes)
    #fname = "R{} 2015-09-06 {:02d}h{:02d}".format(R, hours, minutes)

    #R5 / R6 hoard / R8 low Ajc
    #fname = "R{} 2015-09-14 {:02d}h{:02d}".format(R, hours, minutes) 

    #R1 & R2 nice growgrow / R4 hoard / R8 proper
    #fname = "R{} 2015-09-16 {:02d}h{:02d}".format(R, hours, minutes)

    #Good for R1, R2, R4, R5, R6, R7, R8: 2015-09-17 Evening
    #fname = "R{} 2015-09-17 {:02d}h{:02d}".format(R, hours, minutes)
    fname = "R{} 2015-09-15 {:02d}h{:02d}".format(R, hours, minutes)

    #R1&R2 nice growgrow / R8 proper
    #fname = "R{} 2015-09-18 {:02d}h{:02d}".format(R, hours, minutes)
    #fname = "R{} 2015-09-18 {:02d}h{:02d}".format(R, hours, minutes)

    #fname = "R3 2015-09-18 23h20"
    index, o2data, co2data, pHdata = get_pandas_data(fname)
    pHdata += pH_offset
    env = environment.test_environment()


    cyclestart_ts = find_cyclestart(index, o2data)
    print("Cycle Start: {}".format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cyclestart_ts))))
    a,b = index_from_cyclestart_timestamp(index, cyclestart_ts)
    a = max(0, a-5)
    index = index[a:b]-index[a+5]
    o2data = o2data[a:b]/100
    co2data = co2data[a:b]/100
    n2data = 1 - o2data - co2data

    pHdata.index = pHdata.index.astype(int)/1E9 - cyclestart_ts 

    #Set number of particles that we want to use for each of the M parallel models
    N = 100*[5, 5, 5, 5, 3, 5, 5, 5][R-1] 

    M = 8 #Parallel

    tdiff = 158 #Offgas delay from Headspace to MS

    print("Ready to close (CTRL-C)")

    #Here we can optimize for kLa or timedelay of offgas measurements
    #for tdiff in range(140, 160, 2):
    #for kLa in np.linspace(1.6,2.0,11):
    #for dUR in np.logspace(-9, -7, 10):
    if True:
        pr = "Reactor: {}\tTemp: {}\tdUR: {}\nFlow: {}\tkLa: {}\tI: {}\tpH_offset: {}\n"
        print(pr.format(R, T, dUR, F_in, kLa, I, pH_offset))

        index2 = index + tdiff

        env.set('T',  {'celsius': T})

        Fn_in = F_in/1000/60/env.get('Vm') #mol/s

        env.set('kLa', {'kLaO2': kLa/60})

        env.set('dUR', np.array([0, dUR, dUR]))

        env.set('I', I)
        
        #env.set('P', 1.1)

        #Use ingas signal from measurements, or pick reference values
        env.set('ingas', {'Fn':Fn_in, 'x':[0.790400, 0.209250, 0.000350]})

        #Select the correct partial function - allows you to calculate what the influence is of wrongly estimated paramters e.g. kLa, UR, diffusion
        #extra = partial("-D{:03d}-E{:.2f}".format, tdiff)
        #extra = partial("-kLa{:.2f}-E{:.2f}".format, kLa)
        extra = partial("-dUR{:.2f}-E{:.2f}".format, dUR * 1E8)

        plt.close('all')

        x_h_meas = np.stack([index2, n2data, o2data, co2data], axis=1)

        result = average(N, x_h_meas, pHdata, env, M)
        CURmean = np.convolve(result[:,2], np.ones(10), 'same')/10
        CURstddev = (result[:,2] - CURmean)**2
        CURsums = np.sum(CURstddev) * 1E12

        print(tdiff, kLa, dUR, CURsums)
        extra = extra(CURsums)
        fpathname = 'images/plots/{}-{}{}'.format(fname, time.strftime('%Y%m%d-%H%M'), extra)
        save_fig(fpathname)

    extra = ""
    extra += "-D{:03d}-E{:.2f}".format(tdiff, CURsums)

    fname = 'images/plots/{}-{}{}'.format(fname, time.strftime('%Y%m%d-%H%M'), extra)
    save_fig(fname)
