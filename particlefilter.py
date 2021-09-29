#!/usr/bin/env python
# coding: utf-8

"""Particle Filter of biomass respiration profiles"""
STRICT = False
SEED = 5

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import seed, uniform, randn
import scipy.stats
#from scipy.stats import truncnorm
from multiprocessing import Pool, cpu_count

import time
import datetime
import tempfile

#https://github.com/ipython/ipython/issues/11098)
import plotting # needs to be in a separate cell Jupyter Notebooks
import environment
#from getdata import *

class ParticleFilter:
    def __init__(self, N, env, data, plot_gasnames=None):
        self.N = N
        self.resample_threshold = N/2
        self.resample_threshold = N #Always resample after measurement

        self.environment = env
        self.envt = self.environment(None)  #Initialization Environment 
        self.co2 = self.environment.co2speciation

        self.data = data                    #Contains all online measurments
        self.dataidx = None
        self.verify_and_clean_data()

        #self.est_var = np.zeros((d,self.v)) #TODO: Remove?

        self.w = self.environment.n_gasses  #Number of gasses
        self.v = 4*self.w + 1               #Number of states (UR, c_l, x_b, x_h) + 1 co2l
        self.est_mean = pd.DataFrame(
                np.zeros((len(self.data), self.v))*np.nan, 
                index=self.data.index,
                columns = self.get_column_names())

        self.particles = None
        self.weights = None

        self.plot_gasnames = plot_gasnames or list(self.environment.gasnames)
        self.plot = None
        self.fit_lines = {}

    def bigplot(self):
        h = len(self.plot_gasnames) // 2
        self.plot = self.data[self.plot_gasnames].plot(subplots=True, layout=(h,2), marker='.', ls='', sharex=True)
        new_columns = {gn:'fit' for gn in self.plot_gasnames}
        fitdata = self.data[self.plot_gasnames]*np.nan
        fitdata = fitdata.rename(columns=new_columns)
        fit_plots = dict(zip(self.plot_gasnames, fitdata.plot(subplots=True, marker='x', ax = self.plot, ls='', ms=8, legend=False)))
        for key, subplot in fit_plots.items():
            for line in subplot.get_lines():
                label = line.get_label()
                if label == 'fit': #Skip original 
                    self.fit_lines[key] = line
        #.getlines()
        plt.show(block=False)
        plt.pause(0.1)

    def updateplot(self):
        d = self.est_mean.iloc[self.dataidx]
        e = self.envt
        for key, line in self.fit_lines.items():
            data_ = line.get_ydata()
            data_[self.dataidx] = d['x_h'+key]/e['Pf'] #Plot dry
            line.set_ydata(data_)

        for pl in self.plot:
            for pp in pl:
                pp.relim()
                pp.autoscale_view(True,True,True)

        plt.pause(0.0000001)

    def create_uniform_particles(self, URlow, URhigh, initialstate):
        self.particles = np.empty((self.N, self.v))
        self.particles[:, :self.w] = uniform(URlow, URhigh, size=(self.N, self.w))
        self.particles[:, self.w:] = initialstate
        self.weights = np.ones(self.N)/self.N

    def get_equilibrium_state(self, x_h, pH=None, cCO2hydr=None):
        """Assume equilibrium of reactor with headspace gas"""
        #TODO: Make array-like to allow fast investigation of likely initial state
        e = self.envt
        x_b = x_h
        c_l = e['P'] * e['H'] * x_b
        if self.co2 and (cCO2hydr is None):
            if pH is None:
                raise Exception("Initial state estimate with CO2 requires pH.")
            co2idx = self.environment.idx_by_name['CO2']
            cCO2hydr = self.co2speciation(pH, equilibriateCO2=c_l[co2idx])
        return np.hstack([c_l, x_b, x_h, cCO2hydr])

    def create_shortnames(self):
        self.URs = self.particles[:, :self.w]
        self.c_l = self.particles[:,self.w:2*self.w]
        self.x_b = self.particles[:,2*self.w:3*self.w]
        self.x_h = self.particles[:,3*self.w:4*self.w]
        # Specific views for CO2 speciation
        if self.co2:
            co2idx = self.environment.idx_by_name['CO2']
            self.cCO2 = self.c_l[:,co2idx]
            self.cCO2hydr = self.particles[:,-1]

    def get_column_names(self):
        gns = list(self.environment.gasnames)
        URnames = ["UR_{}".format(gn) for gn in gns]
        c_lnames = ["c_l{}".format(gn) for gn in gns]
        x_bnames = ["x_b{}".format(gn) for gn in gns]
        x_hnames = ["x_h{}".format(gn) for gn in gns]
        return URnames+c_lnames+x_bnames+x_hnames+['CO2hydr']

    def pre_predict(self, steps=1):
        """Change uptake rate UR with noisy dUR, taking into account the covariance

        the covariance matrix must be square [wxw] with w the number of gas compounds
        dUR can be either a vector of length w or a scalar and is in unit mol/s
        """
        e = self.envt
        mean = np.zeros(self.w)
        URcov = np.multiply(e['URcov'], steps*e['dUR']**2)
        self.URs += np.random.multivariate_normal(mean, URcov, size=self.N)

    def exchange_liquid(self):
        e = self.envt #Shortcut name
        feedprofiles = [e[fn] for fn in ('FeedW', 'FeedN', 'FeedC') if fn in e]

        e['Vl_old'] = e['Vl']
        e['Vb_old'] = e['Vb']
        e['Vh_old'] = e['Vh']

        net_flow_rate = 0 #L/s

        for flow_rate, c_l_in in feedprofiles:
            #if 'FeedW' in e:
            #    self.URs[0] *= 0 #Fix CO2 UR to 0
            net_flow_rate += flow_rate
            #Mass balance
            self.c_l *= e['Vl'] / (e['Vl'] + flow_rate) #Dilute
            self.cCO2hydr *= e['Vl'] / (e['Vl'] + flow_rate) #Dilute
            self.c_l += (c_l_in * flow_rate) / (e['Vl'] + flow_rate) #Add
            #Assume no CO2 hydr flows in
            #self.c_l[:] = (self.c_l * e['Vl'] + flow_rate * c_l_in) / (e['Vl'] + flow_rate)
            e['Vl'] += flow_rate

            print("Liquid {:.2f}, Bubble {:.2f}, Headspace {:.2f}, SUMx_h {:.2f}".format(e['Vl'], e['Vb'], e['Vh'], PF.x_h.sum(axis=1).max()/e['Pf']))

        if "Effluent" in e: 
            e['Vl'] += e['Effluent'][0]
            net_flow_rate += e['Effluent'][0]
            #Reduce UptakeRates by removed fraction
            f = e['Vl']/e['Vl_old']
            self.URs *= f
            self.URs[:,0] /= f
            print("Liquid {:.2f}, Bubble {:.2f}, Headspace {:.2f}, SUMx_h {:.2f}".format(e['Vl'], e['Vb'], e['Vh'], PF.x_h.sum(axis=1).max()/e['Pf']))

        e['Vb'] = 0.05 * e['Vl'] #TODO: replace magic number for gas holdup
        e['Vh'] = e['Vr'] - e['Vl'] - e['Vb']

        #if 'FeedW' in e:
            #import pdb; pdb.set_trace()

        e['Fnh2o'] = net_flow_rate/e['Vm'] #Wet gas flow

        ##NOTE: SHOULD NOT HAPPEN
        if e['Fnh2o'] + e['ingas']['Fwet'] < 0: #effluent faster than ingas flowrate
            print("WARNING: Effluent rate is higher than gas influent rate")
            #This initally reduces internal pressure and eventually pulls in gas
            #x_g_in = in_comp

        if net_flow_rate != 0:
            #Store volumes to environment #TODO: better to store Vl in PF?
            self.environment.set('Vl', e['Vl'])
            self.environment.set('Vb', e['Vb'])
            self.environment.set('Vh', e['Vh'])


    def predict(self):
        """Proliferate accord*ing to physicochemical model."""
        e = self.envt #Shortcut name
        self.exchange_liquid()

        #Bulk <-> gas-bubbles [mol/sec]
        TRs = e['Vl'] * e['kLa'] * (e['P'] * e['H'] * self.x_b - self.c_l) 
        #TODO: Think about additional gas coming in through exchange resulting in
        #      a temporary pressure drop/increase.

        #Gasflow from liquid to headspace
        Fnb2hs = (e['ingas']['Fwet'] + 
                  e['recgas']['Fwet'] -
                  TRs.sum(axis=1)/e['Pf'])[:,None] #[mol/sec]
        
        #Changes in liquid, bulk gas, and headspace gas composition
        dc_ls = (TRs - self.URs)/e['Vl']    # [mol/sec]
        dx_bs = self.x_b * (e['Vb_old']/e['Vb'] - 1) +\
                ((e['ingas']['Fwet'] * e['ingas']['xwet'] +
                  e['recgas']['Fwet'] * self.x_h) - 
                 self.x_b * Fnb2hs - TRs) / (e['Vb']/e['Vm'])

        #dx_hs = Fnb2hs * (self.x_b - self.x_h) / (e['Vh']/e['Vm'])

        #Compensate for additional inflow/outflow of liquids
        dx_hs = self.x_h * (e['Vh_old']/e['Vh'] - 1) +\
                (Fnb2hs * self.x_b -
                (Fnb2hs + e['Fnh2o']) * self.x_h) / (e['Vh']/e['Vm'])

        if self.co2: #Perform CO2 speciation when present
            self.co2speciation()

        self.c_l += dc_ls
        self.x_b += dx_bs
        self.x_h += dx_hs

        #QC - relatively expensive
        if STRICT:
            self.remove_impossible_states()

    def remove_impossible_states(self):
        LOWLIQC = -1E-3 #[1 mmol error band]
        self.x_b[np.any(self.x_b>1, axis=1)] = np.nan         #Gasses cannot be over 100%
        self.x_b[np.any(self.x_b<0, axis=1)] = np.nan         #And not under 0% (margin)
        self.x_h[np.any(self.x_h>1, axis=1)] = np.nan         #Gasses cannot be over 100%
        self.x_h[np.any(self.x_h<0, axis=1)] = np.nan         #And not under 0% (margin)
        self.c_l[np.any(self.c_l<LOWLIQC, axis=1)] = np.nan   #Dissolved gasses cannot be under 0%

    def co2speciation(self, pH=None, equilibriateCO2=None):
        if pH is None:
            pH = self.data.iloc[self.dataidx]['pH']
        e = self.envt #Shortcut names
        kK = e['co2constants']

        # CO2 speciation input: environment e, cCO2hydr liquid, pH
        aH  = 10**-pH                
        aOH = e['Kw']/aH                  
        cH  = aH / e['ions']['H'].gamma       # [H+]  
        #cOH = aOH / compounds['OH']/gamma     # [OH-] 

        #Activity correction for equilibrium constants
        Ka1 = kK['Ka1']/(e['ions']['H'].gamma * e['ions']['HCO3'].gamma)
        Ka2 = kK['Ka2'] * e['ions']['HCO3'].gamma/(e['ions']['H'].gamma * e['ions']['CO3'].gamma)
        fH2CO3 = 1/(1+Ka1/cH+Ka1*Ka2/cH**2)  # Equilibrium [H2CO3]
        fHCO3  = 1/(1+cH/Ka1+Ka2/cH)         # Equilibrium [HCO3]
        fCO3   = 1/(1+cH/Ka2+cH**2/Ka1)      # Equilibrium [CO3]

        #NOTE: Sum of CO2 fractions should be 1 (or mathematically close < 1E-6)
        if STRICT: 
            np.testing.assert_allclose(fH2CO3+fHCO3+fCO3, 1, rtol=1E-6)

        p1 = (kK['k1']  * e['cH2O'] + kK['k2'] * aOH) # CO2(aq) -> CO2hydr
        p2 = (kK['k_1'] * fH2CO3 - kK['k_2'] * fHCO3 * e['ions']['HCO3'].gamma) #CO2hydr -> CO2(aq)

        if equilibriateCO2 is None:
            dcCO2hydrs = self.cCO2 * p1 - self.cCO2hydr * p2
            self.cCO2hydr += dcCO2hydrs
            self.cCO2 -= dcCO2hydrs
        else:
            return equilibriateCO2*p1/p2 #Equilibrium concentration of CO2hydr

    def getpH(self, t=0):
        #TODO: Retrieve pH at timepoint t from the dataset, what about initial pH
        return self.data.iloc[t]['pH']

    def update_weights(self, meas):
        #get measurement from timestamp t (with sensor std_err R)
        #measurement consists of x_h (dry) measurement [N2, O2, CO2]
        #We do not have a gas measurement for each timestamp
        #now we can only use the DO measurements TODO
        logpdf = scipy.stats.norm(self.x_h - meas, self.envt['meas_stddev']*20).logpdf(0)
        logpdf = (logpdf - logpdf.max(axis=0)).sum(axis=1)
        mask = logpdf>-300
        logpdf[~mask] = -np.inf
        pdf = np.exp(logpdf) 

        self.weights *= pdf
        #if np.nan in self.weights or 0 in self.weights:
            #import pdb; pdb.set_trace()
        self.weights /= sum(self.weights)

    def setup(self, x_h0=None):
        e = self.envt
        gn = self.environment.gasnames[0]
        if x_h0 is True:
            fi = self.data[gn].first_valid_index()
            iloc = data.index.get_loc(fi)
            x_h0 = data.iloc[iloc][list(self.environment.gasnames)]*e['Pf']
        else:
            x_h0 = e['ingas']['xwet']

        initialstate = self.get_equilibrium_state(x_h0, self.getpH())

        #UR0low  = np.array([0.7, 0, 0, 0, -8, -0.05])*1E-7 #R1
        #UR0high = np.array([0.7, 0, 0, 0, -6, -0.00])*1E-7 #R1

        UR0low  = np.array([0.3, 0, 0, 0, -4, -0.035])*1E-7 #R2
        UR0high = np.array([0.3, 0, 0, 0, -3, -0.03])*1E-7 #R2

        #Maxim lege reactor
        #UR0low  = np.array([0, 0, 0, 0, -0, -0])*1E-7 #R1
        #UR0high = np.array([0, 0, 0, 0, -0, -0])*1E-7 #R1
        self.create_uniform_particles(UR0low, UR0high, initialstate)
        self.create_shortnames()
        #get data
        #initialize environment 
        #create particles
        #shortnames
        #graphing
        pass

    def estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var  = np.average((self.particles - mean)**2, weights=self.weights, axis=0)
        return mean, var

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def resample(self):
        indexes = systematic_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        self.weights.resize(self.N)
        self.weights.fill(1.0 / self.N) #Hard reset of all weights

    def run(self):
        #prepare plot
        self.bigplot()
        #if fitplotfunc:
        #    fitplotfunc = fitplotfunc("TODO", data_ts, data_meas)
        #    fitplotfunc.send(None) #This plots the raw data

        try:
            for self.dataidx, (index, datarow) in enumerate(data.iterrows()):
                e = self.envt = self.environment(index) 
                g = datarow[list(self.environment.gasnames)]
                self.predict()

                if g.any(): #Offgas measurement available for this timepoint
                    self.update_weights(list(g*e['Pf']))
                    self.est_mean.iloc[self.dataidx] = self.estimate()[0]
                    self.updateplot()

                    if self.neff() < self.resample_threshold: #Resample 
                        self.resample()

                self.pre_predict()

                #if fitplotfunc:
                #    meani, vari = np.copy(self.est_mean[i]), np.copy(est_var[i])
                #    meani[3*w:4*w] /= e['Pf'] #Represent the head space fraction as dry gas
                #    vari[3*w:4*w] /= e['Pf']
                #    fitplotfunc.send((data_t, meani, vari))

        except Exception as e:
            import traceback
            traceback.print_exc()

    def verify_and_clean_data(self):
        gn = list(self.environment.gasnames)
        mean = self.data[gn].sum(axis=1, skipna=False).mean()
        ctn = environment._close_to_n
        if not(ctn(mean, 1) or ctn(mean, 100)):
            raise Exception("Offgas measurement data does not sum op to 100%")

        #normalize
        self.data[gn]/= np.round(mean,0) 

        cols = ['pH']
        if 'DO' in self.data.columns:
            cols.append('DO')
        
        if self.data[cols].isna().any(axis=None):
            self.data[cols] = self.data[cols].interpolate()


def get_test_data():
    #ret =   pd.read_excel('/Users/gerben/Downloads/maximcycle.xlsx', 
    #ret =   pd.read_excel('/Users/gerben/Downloads/20202809.2300.MAXIMR1.xlsx', 
    #ret =   pd.read_excel('/Users/gerben/Downloads/maximcycle_tiny.xlsx', 
    #ret =   pd.read_excel('/Users/gerben/Downloads/20202809.2300.MAXIMR2.xlsx', 
    #ret =   pd.read_excel('/Users/gerben/Downloads/emptycycles.xlsx',
    #        parse_dates=True,
    #        index_col='datetime')
    #ret =   pd.read_csv('/Users/gerben/Downloads/21daysR1.csv',
    ret =   pd.read_csv('/Users/gerben/Downloads/21daysR2.csv',
            parse_dates=True,
            index_col='datetime')
    ret.index = np.array((ret.index.astype(int)/1E9).astype(int))
    return ret

def test_environment():
    #maxim
    env = environment.Environment(('CO2', 'N2', 'O2', 'Ar', 'H2', 'CH4'))

    Vr      = 2.2           # [L]     Total Reactor Volume
    Vl      = 1.00          # [L]     Working volume
    Vb      = 0.05*Vl       # [L]     Bubble volume
    Vh      = Vr - Vl - Vb  # [L]     Headspace volume
    P       = 1.1           # [Bar]   Pressure

    env.set('Vr', Vr)
    env.set('Vl', Vl)
    env.set('Vb', Vb)
    env.set('Vh', Vh)
    env.set('P', P)

    Tc = 34                 # [C]     Temperature in C
    env.set('T', {'celsius':Tc})

    env.set('I', 0.075) #PHB system

    env.set('kLa', {'kLaO2': 0.15/60}) #OLd

    #F_in = 46.6 #ml/min #R1
    F_in = 56 #ml/min #R2
    Fn_in = F_in/1000/60/env.get('Vm') #mol/s
    #env.set('ingas', {'Fn':Fn_in, 
    #    'x':[0.0465597,  #CO2
    #         0.9525260,  #N2
    #         0.0002418,  #O2
    #         0.0001847,  #Ar
    #         0.0001038,  2H2
    #         0.0003789   #CH4

    env.set('ingas', {'Fn':Fn_in, 
       'x':[0.0465600,  #CO2
            0.9525000,  #N2
            0.0003500,  #O2
            0.0001100,  #Ar
            0.0001000,  #H2
            0.0003800   #CH4
       ]})

    #Ingas after new calibration Jan 2021
    #env.set('ingas', {'Fn':Fn_in, 
    #   'x':[0.049715,  #CO2
    #        0.949158,  #N2
    #        0.000809,  #O2
    #        0.000097,  #Ar
    #        0.000021,  #H2
    #        0.000199   #CH4
    #   ]})

    F_rec = 360 #ml/min
    Fn_rec = F_rec/1000/60/env.get('Vm') #mol/s
    env.set('recgas', {'Fn':Fn_rec, 'x':None, 'dry':False})

    #TODO: smartn up Measurement stddev, smooth results
    #                                    CO2     N2     O2     Ar     H2    CH4
    env.set('meas_stddev',  np.array([  10.0,  100.0, 100.0, 100.0,  10.0,   1.0])/2E5)
    #env.set('dUR',          np.array([   0.0,   0.0,   0.0,   0.0,   0.0,   0.0])/1E9)
    #env.set('dUR',          np.array([   1.0,   0.0,   0.0,   0.0,   2.0,   0.20])/1E9) #R1
    #env.set('dUR',          np.array([   2.0,   0.0,   0.0,   0.0,   2.0,   0.20])/1E9) #R1-21days
    #env.set('dUR',          np.array([   0.5,   0.0,   0.0,   0.0,   5.0,   0.10])/1E9) #R2
    env.set('dUR',          np.array([   0.5,   0.0,   0.0,   0.0,   5.0,   0.10])/1E9) #R2-21days
    env.set('URcov',        np.diag( [   1.0,   1.0,   1.0,   1.0,   1.0,   1.0]))

    return env

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

def flow_control(env, name, t):
    effl_flow = -0.500/(12*60) #L/sec
    feedN_flow = 0.05/(1*60) #L/sec
    feedW_flow = 0.400/(4*60) #L/sec
    feedC_flow = 0.05/(1*60) #L/sec
    effl_x_g_in = (0.000415,
                   0.780840,
                   0.209460,
                   0.009340,
                   0.000000,
                   0.000002) #Normal air conditions

    feedN_c_l_in = (0.000000000,
                    0.000688451,
                    0.000000000,
                    0.000000000,
                    0.000000000,
                    0.000000000) #@21C

    feedW_c_l_in = (0.000000000, 
                    0.000688451,
                    0.000000000,
                    0.000000000,
                    0.000000000,
                    0.000000000) #@21C

    feedC_c_l_in = (0.000415000,
                    0.000537570,
                    0.000271616,
                    0.000014700,
                    0.000000000,
                    0.000000003) #@21C
    
    dd = {'Effluent': (effl_flow, np.array(effl_x_g_in)),
          'FeedW': (feedW_flow, np.array(feedW_c_l_in)),
          'FeedN': (feedN_flow, np.array(feedN_c_l_in)),
          'FeedC': (feedC_flow, np.array(feedC_c_l_in))}
    return {name: dd[name]}

def new_cycle(env, name, t):
    print("NEW CYCLE PLEASE: ", t)
    t_effl_start = t + 1
    t_effl_stop = t_effl_start + 12*60

    t_feedW_start = t_effl_start + 1500
    t_feedW_stop = t_feedW_start + 4*60
    t_feedN_start = t_feedW_start + 120
    t_feedN_stop = t_feedN_start + 60

    t_feedC_start = t_effl_start + 60*60
    t_feedC_stop = t_feedC_start + 60

    env.register("Effluent", t_effl_start, t_effl_stop, flow_control)
    env.register("FeedW", t_feedW_start, t_feedW_stop, flow_control)
    env.register("FeedC", t_feedC_start, t_feedC_stop, flow_control)
    env.register("FeedN", t_feedN_start, t_feedN_stop, flow_control)

    tt = t + 12*60*60
    env.register(name, tt, tt+1, new_cycle)


def remove_cleaning(data):
    n = 7200 #two hours of data for cleaning
    numbers = list(data.loc[data[['O2', 'Ar']].sum(axis=1) > 0.2, ['O2', 'Ar']].index) #Max 0.2% of gas O2 and Ar
    nnumbers = np.array(numbers)
    clusters = pd.DataFrame({
        'numbers': numbers,
        'segment': np.cumsum([0] + list(1*(nnumbers[1:] - nnumbers[0:-1] > n))) + 1
        }).groupby('segment').agg({'numbers': set}).to_dict()['numbers']

    datasets = []
    t0 = None
    for v in clusters.values():
        sv = sorted(v)
        a, b = sv[0], sv[-1]
        if b-a < 1000: #skip
            continue
        datasets.append(data.loc[t0:a-1])
        t0 = b

    #data.loc[data[['O2', 'Ar']].sum() > 0.1, gn] = np.nan #REMOVE BAD DATAPOINTS
    #data.loc[data[plot_gn].sum(axis=1) < 99, gn] = np.nan #REMOVE BAD DATAPOINTS
    return datasets

if __name__ == "__main__":
    env = test_environment()

    data = get_test_data()
    #Remove bad data from 21-
    day = 60*60*24
    t_remove = 10*day
    #data = data.iloc[:t_remove]
    #t_remove_end = int(18*day)
    #data = data.iloc[t_remove_end:]
    #qq = np.arange(t_remove, t_remove_end)
    #data.drop(data.index[qq], inplace=True)

    #mpH = 6.85 #R1
    #data['pH'][data['pH']>mpH]=mpH #Fix the pH data to maxpH

    #TODO: Gasdata delay due to tubing volume ~60-100 mL / gasflow 
    #import pdb; pdb.set_trace()
    #data[list(env.gasnames)] = data[list(env.gasnames)].shift(-int(60*0.1/0.0466)) #R1
    data[list(env.gasnames)] = data[list(env.gasnames)].shift(-int(60*0.1/0.056)) #R2
    #data = get_test_data()

    gn = list(env.gasnames)
    data_sets = remove_cleaning(data)
    #t_effl_start = datetime.datetime(2020, 9, 28, 23, 00, 0).timestamp() + 7200 #R1
    #t_effl_start = datetime.datetime(2020, 9, 28, 22, 45, 0).timestamp() + 7200 #R2 #looks worse than starting at 23:00
    #t_effl_start = datetime.datetime(2021, 1, 26, 15, 31, 0).timestamp() + 7200 #R1 empty
    #t_effl_start = datetime.datetime(2021, 1, 26, 15, 31, 0).timestamp() + 3600 #R1 empty

    #SET STARTTIME OF CYCLE
    #t_offset = t_remove_end #R1
    for data in data_sets:
        #TODO DO NOT PARSE SMALL DATASETS
        t_offset = 0
        #t_effl_start = datetime.time(11,0,0) #R1
        cycle_start = datetime.time(10,45,0) #R2
        UTCoffset = datetime.timedelta(hours=2)
        CL = datetime.timedelta(hours=12)
        data_start_date = datetime.datetime.utcfromtimestamp(data.index[0]) #UTCoffset crap
        dt_effl_start = data_start_date.replace(hour=cycle_start.hour, minute=cycle_start.minute, second = cycle_start.second) + UTCoffset
        if data_start_date > dt_effl_start: #Add 1 cycle to initial starttime
            dt_effl_start += CL
        t_effl_start = dt_effl_start.timestamp()
        #t_effl_start = datetime.datetime(2020, 9, 7, 23, 0, 0).timestamp() + 7200 + t_offset#R1 21-days
        #t_effl_start = datetime.datetime(2020, 9, 7, 22, 45, 0).timestamp() + 7200 + t_offset#R2 21-days
        new_cycle(env, "newCycle", t_effl_start-1)
        
        plot_gn=['CO2', 'N2', 'H2', 'CH4']
        PF = ParticleFilter(10000, env, data=data, plot_gasnames=plot_gn)
        PF.setup(x_h0=True)

        #PF.cCO2hydr[:] = 0.0027995
        #PF.cCO2hydr[:] = 0.004
        #gn = PF.environment.gasnames
        #PF.c_l[:, gn.index('CO2')] = 0.0006326 #CO2
        #PF.c_l[:, gn.index('H2')] = 0.0000232 #H2
        #PF.c_l[:, gn.index('N2')] = 0.0006077 #N2

        PF.run()
        PF.est_mean[['UR_CO2', 'UR_N2', 'UR_H2', 'UR_CH4']].dropna().plot(secondary_y=True, subplots=True, marker='.', ax=PF.plot, ls='--', color='black', alpha=0.5, legend=False)
        #PF.est_mean[['UR_CO2', 'UR_N2', 'UR_O2', 'UR_Ar', 'UR_H2', 'UR_CH4']].dropna().plot(secondary_y=True, subplots=True, marker='.', ax=PF.plot, ls='--', color='black', alpha=0.5, legend=False)
        plt.pause(0.1)
        aa = PF.est_mean.dropna()
        aa.loc[:, gn] = data[gn].dropna()
        aa.loc[:, 'Pf'] = env.get('Pf')
        _, fn = tempfile.mkstemp(prefix='maxim-', suffix='.csv')
        print(fn)
        aa.to_csv(fn)
    import sys; sys.exit(0)




#def get_args(N, data, softner=20, itern=4):
#    #Gas properties:
#    #                    N2    O2     CO2
#    UR0low  = np.array([0,    0E-6, -1E-6])
#    UR0high = np.array([0,    1E-6, -0E-6])
#    dUR     = np.array([0,    1E-7,  1E-7])
#    R       = np.array([400,    60,    20])/1E6*softner #Measurement stddev, smooth results
#    cov     = np.diag( [0.,    1.,     1.  ])   #Universal: assume no correllations
#    #cov = [[0,0,0],[0,1,-0.2],[0,-0.2,1]]
#
#    #A single run with optimized parameters
#    TITLE = "Particle Filter (N={})".format(N)
#    for _ in range(int(itern)):
#        yield [N, UR0low, UR0high, cov, dUR, data, R,  None, TITLE, plotting.plot_fit]
#        np.random.random(N) 
#
#    ##Impact of covariance 
#    #for i in range(0,11):
#    #    cov = np.array(np.diag([1.,1.,1.]))
#    #    cov[1,2] = cov[2,1] = -0.1*i
#    #    TITLE = "Particle Filter (N={}) with cov OUR-CUR {:.1f})".format(N, -0.1*i)
#    #    yield [N, UR0low, UR0high, cov, dUR, x_h_meas, R,  mi, None, TITLE, plot_fit]
#
#    ##Impact of dynamics in Uptake Rates dUR
#    #for i in range(1,8):
#    #    dCUR = (4+i)*1E-8
#    #    dUR = np.array([0, 6E-8, dCUR])
#    #    TITLE = "Particle Filter (N={}) with dCUR ({:.1e} mol/sec)".format(N, dCUR)
#    #    yield [N, UR0low, UR0high, cov, dUR, x_h_meas, R,  mi, None, TITLE, plot_fit]
#
#    ##Impact of measurement interval on solution
#    #for i in range(1,8):
#    #    mi = 60 * i
#    #    TITLE = "Particle Filter (N={}) with measurement interval ({} sec)".format(N, mi)
#    #    yield [N, UR0low, UR0high, cov, dUR, x_h_meas, R,  mi, None, TITLE, plot_fit]
#
#
#    ##Detect best smoothing N2 measurement error 
#    #for i in range(0,7):
#    #    mme = int(10**i)
#    #    TITLE = "Particle Filter (N={}) with N2 measurement error ({} ppm)".format(N, mme)
#    #    R = np.array([mme, 300, 300])/1E6
#    #    yield [N, UR0low, UR0high, cov, dUR, x_h_meas, R,  mi, None, TITLE, plot_fit]
#
#def low_high_ppm(N, data, ppm, softner=20):
#    """Calculate the lower and higher bound solution by allowing an error of %ppm in CO2-O2"""
#    #Gas properties:
#    #                    N2    O2     CO2
#    #UR0low  = np.array([0,    2E-6, -4E-6])
#    #UR0high = np.array([0,    4E-6, -2E-6])
#    UR0low  = np.array([0,    0E-6, -1E-6])
#    UR0high = np.array([0,    1E-6,  0E-6])
#    dUR     = np.array([0,    8E-8,  8E-8])
#
#    #Measurement error on the MS is approximately 165 ppm for N2 and O2, and 17 ppm for CO2
#    #To allow a significant fraction of particles to remain, we aim for a cubic root of N 
#    #R       = np.array([8000, 300,   300])/1E6  #Measurement stddev, smooth results
#    #20191003-0900
#    R       = np.array([400, 60, 20])*1e-6*softner #measurement stddev, smooth results
#    #20191103-0900
#    #R       = np.array([165, 163, 17])*1e-6*softner #measurement stddev, smooth results
#    cov     = np.diag( [1.,    1.,     1.  ])   #Universal: assume no correllations
#    #cov = [[0,0,0],[0,1,-0.2],[0,-0.2,1]]
#
#    #plot_fit = None #If no intermediate plots are required (Speed)
#
#    offsets = np.array([[0,     0,   0,    0],
#                        [0,  ppm, -ppm,    0],
#                        [0, -ppm,  ppm,    0],
#                        [0,  ppm,    0, -ppm],
#                        [0, -ppm,    0,  ppm]])
#
#    env = environment.test_environment()
#
#    for offset in offsets:
#        TITLE = "PF (N={}) O2 {:+} ppm / CO2 {:+} ppm)".format(N, offset[2], offset[3])
#        shifteddata = data+offset*1E-6
#        sdd = shifteddata[:, 1:]
#        sdd[sdd<0] = 0 #Make sure that no measurement are negative
#        yield [N, env, UR0low, UR0high, cov, dUR, shifteddata, R,  None, TITLE, plotting.plot_fit]
#        #Ensures that parallel fits start at a new randomization point
#        np.random.random(N) 
#
#def average(N, data, parallel=8):
#    soft = 5
#
#    # If we can unload one core during parallel computing 
#    # without sacrificing calculation time do that
#    ncpu = cpu_count()
#    poolsize = ncpu - (np.ceil(parallel/(ncpu-1)) == np.ceil(parallel/ncpu))
#
#    with Pool(poolsize) as p:
#        ans = np.stack(p.starmap(runPF, get_args(N, data, soft, parallel)))
#
#    meas = ans[:,0]
#    mean = np.average(meas, axis=0)
#    mean[:,9:12]*=(100/Pf) #Make dry air and convert fraction to percentage
#    stddev = np.std(meas, axis=0)
#
#    dof = parallel - 1
#    conf95 = scipy.stats.t.ppf(0.975, dof)/np.sqrt(dof) #double sided
#
#    hv = mean + stddev * conf95
#    lv = mean - stddev * conf95
#    xh_lv = lv[:,9:12]
#    xh_hv = hv[:,9:12]
#    UR_lv = lv[:,0:3]
#    UR_hv = hv[:,0:3]
#    UR_mv = mean[:,0:3]
#
#    TITLE = "PF (N={}) (s={}x{})".format(N, parallel, soft)
#    plot = plotting.plot_fit(TITLE, data[:,0]/3600, data[:,1:4]*100, xh_lv, xh_hv, UR_mv, UR_lv, UR_hv)
#    plot.send(None) #Draw the plot
#
#
#def low_high():
#    N = int(1E4) #number of particles to generate
#    ppm = 200      #measurement offset eror of mass spec
#    soft = int(np.sqrt(N)/2) #smoothing factor (arbitrary unit)
#    soft = 10
#
#    data = x_h_meas
#    with Pool(5) as p:
#        ans = np.stack(p.starmap(runPF, low_high_ppm(N, data, ppm, soft)))
#
#    meas = ans[:,0]
#    meas[:,:,9:12]/=Pf #Make dry air
#    stddev = np.sqrt(ans[:,1])
#    stddev[:,:,9:12]/=Pf #Make stddev also for dry air
#    hv = meas + stddev
#    lv = meas - stddev
#    xh_lv = np.min(lv[:,:,9:12], axis=0)
#    xh_hv = np.max(hv[:,:,9:12], axis=0)
#    UR_lv = np.min(lv[:,:,1:4], axis=0)
#    UR_hv = np.max(hv[:,:,1:4], axis=0)
#    UR_mv = np.average(meas[:,:,1:4], axis=0)
#
#    TITLE = "PF (N={}) (e={}ppm) (s={})".format(N, ppm, soft)
#    plot_fit(TITLE, data[:,0]/3600, data[:,1:4], xh_lv, xh_hv, UR_mv, UR_lv, UR_hv)
#
#    plt.show(block=False)
#
#def test():
#    args = low_high_ppm(1000, x_h_meas, 500).send(None)
#    print(runPF(*args))
#
#def get_pandas_data():
#    import pandas
#    #%matplotlib
#    fname = '/Users/gerben/Downloads/12693359/cycledata/R3/R3 2015-09-18 11h20.h5.db'
#    with pandas.HDFStore(fname, 'r', complib='blosc') as store:
#        d = store.get('data')
#    CO2data = d['CO2'][::180]
#    O2data = d['O2'][::180]
#    index = d.index[::180].astype(int)/1E9
#
#
#
#
#if __name__ == "__main__":
#    seed(SEED)
#    STRICT and np.seterr(all='raise')
#
#    #filepathname = "./gasdata/10300004.csv"
#    filepathname = "./gasdata/11030000.csv"
#    index, o2data, co2data = loadGas(filepathname) #Index in seconds since epoch
#    cyclestart_ts = find_cyclestart(index, o2data)
#    print("Cycle Start: {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cyclestart_ts))))
#    a,b = index_from_cyclestart_timestamp(index, cyclestart_ts)
#    a = max(0, a-5)
#    index = index[a:b]-index[a+5]
#    o2data = o2data[a:b]/100
#    co2data = co2data[a:b]/100
#    n2data = 1 - o2data - co2data
#    #tmax, tmin = getFeastFromO2gas(index, o2data, 0.01)
#    #x,y = getFeastEndFromCO2gas(index, co2data)
#    #feastlength = (x-tmin)
#    #print("Feastlenth: {:.0f} sec".format(feastlength))
#    x_h_meas = np.stack([index, n2data, o2data, co2data], axis=1)
#
#    #Load raw data 
#    #fname = '20191003-0900'
#    fname = '20191103-0900'
#    #fname = 'demoxh_full'
#    #x_h_meas = np.genfromtxt('demoxh_full.csv', delimiter=",")
#    #x_h_meas = np.genfromtxt('data/{}.csv'.format(fname), delimiter=",", skip_header=1)
#    
#    #runPF(*args)
#    print("Ready to close (CTRL-C)")
#
#    N = int(1E3) #number of particles to generate
#    #average(N, x_h_meas, 20)
#    #low_high()
#    test()
#    #with Pool(7) as p:
#        #ans = p.starmap(runPF, get_args())
#
#    fname = 'images/plots/{}-{}'.format(fname, time.strftime('%Y%m%d-%H%M'))
#    print("fname: {}".format(fname))
#    import pdb; pdb.set_trace()
#    plt.savefig(fname+".png", dpi=300)
