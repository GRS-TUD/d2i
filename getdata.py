# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import rdp
import plotting #setup the plotting styles (needs to be in a separate cell: https://github.com/ipython/ipython/issues/11098)

################################################################################
################## COLLECTION OF Feast/Famine HELPER FUNCTION ##################
################################################################################

def makeRDP(index, data, epsilon):
    #Ramer Douglas Peucker algorithm for curve simplification
    zipped = list(zip(index, data))
    return np.array(rdp.rdp(zipped, epsilon=epsilon))

def RDPplot(index, data, ylim=(None,None), epsilons=[0, 0.02], ax=None):
    if ax is None:
        _, ax = plt.subplots()
    data_rdp = {e: makeRDP(index, data, e) for e in epsilons}
    ax.scatter(index, data, marker='+', s=2)
    for k,v in data_rdp.items():
        ax.plot(v[:,0], v[:,1], label="$\epsilon=${:.3f}, l={}".format(k, len(v)), alpha=0.5)
    ax.legend(loc=4, numpoints=1)
    ax.set_ylim(*ylim)

def getFeastFromO2gas(index, data, e = 0.15):
    #We might even be able to just use the CO2 data, by looking at the max CO2 peak. #TODO
    #We should have the starttime already
    RDP = makeRDP(index, data, e)
    di = np.diff(RDP, axis=0)
    slopes = np.divide(di[:,1],di[:,0])
    tmin = RDP[np.argmin(slopes),0]
    tmax = RDP[np.argmax(slopes),0]
    return tmax,tmin

def getFeastEndFromCO2gas(index, data, e = 0.15):
    #What should we do if there are a couple of high CO2 peaks (e.g. at higher pH)
    #Get the steapest CO2 slopes. If the first measurement behind the peak is the steapest, than 'peak' measurement is 
    #likely measured on the way down. If the steapest segment is the 2nd after the peak, than the peak is likely after the 'peak'
    #therefore we can do an intersect calculation based on this knowledge.
    #===
    #Find CO2 peak (end-feast, check with O2)
    #Get slope of 2 segments directly following it .
    #The steapest (negative) slope is the guide (N), and intersect it with the slope of segment (N-2)
    idxmax = data.argmax()+10
    RDP = makeRDP(index[:idxmax], data[:idxmax], e) #Is the data is [0-1]% or [0-100]%
    di = np.diff(RDP, axis=0)
    slopes = np.divide(di[:,1],di[:,0])
    aggr = np.argmin(slopes)
    return get_intersect(RDP[aggr,:], RDP[aggr+1,:], RDP[aggr-2,:], RDP[aggr-1,:])


def reject_outliers(data, m=1):
    return np.mean(data[abs(data - np.mean(data)) <= m * np.std(data)])

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

str2ts = lambda x: datetime.strptime(x.decode('utf-8').split('.')[0], '%Y-%m-%d %H:%M:%S').timestamp()

def loadGas(fpname):
    # data = np.genfromtxt("./gasdata/10300004.csv", delimiter=";", skip_header=1, converters={0:str2ts})
    data = np.genfromtxt(fpname, delimiter=";", skip_header=1, converters={0:str2ts})  
    index, data  = data[:,0], data[:,1:5]  #CO2 = data[:,1:], O2 = data[:,3:]

    # Group offgas data measurements
    idx = 0
    blocks = []
    for e,i in enumerate(np.diff(index)):
        if i > 20: #new measurement block
            blocks.append((idx, e+1))
            idx = e+1
            continue

    co2data = np.array([reject_outliers(data[b[0]:b[1],0]) for b in blocks])
    o2data = np.array([reject_outliers(data[b[0]:b[1],2]) for b in blocks])
    n2data = 100 - o2data - co2data

    index_b = np.array([np.mean(index[b[0]:b[1]]) for b in blocks])
            
    return index_b, o2data, co2data

def find_cyclestart(index, o2data, CL=12):
    # Take the first cyclelength [CL=12] hours of the dataset
    cyclelength = CL*60*60
    ie = np.searchsorted(index, index[0]+cyclelength, side='right')
    tmax, tmin = getFeastFromO2gas(index[:ie], o2data[:ie], 0.03)

    #find closest quarter hour for start time
    quarterhour = 60*15
    d, m = divmod(tmin, quarterhour)
    if m > quarterhour/2: #flip left or right
        d+=1
    d *= quarterhour
    return d

def index_from_cyclestart_timestamp(index, timestamp, CL=12):
    cyclelength = CL*60*60
    istart = np.searchsorted(index, timestamp, side='left')
    iend = np.searchsorted(index, timestamp+cyclelength, side='right')
    return istart, iend
