# https://jonchar.net/notebooks/matplotlib-styling/
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

def _isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

#SOME SETUP FUNCTIONS FOR PLOTTING
def setup():
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

    # Set the font used for MathJax - more on this later
    rc('mathtext',**{'default':'regular'})

    #86mm vs 178mm (half page vs full page)
    x = 86/10
    #y = x/4*3
    y = x/6*3

    if _isnotebook():
        dpi = 300
    else:
        dpi = 100

    rc('figure', **{'figsize':[x,y], 'dpi':dpi})

#line width 0.25 - 1 pt.

def stylize_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
     
    ax.xaxis.set_tick_params(top=False, direction='out', width=1)
    ax.yaxis.set_tick_params(right=False, direction='out', width=1)

################################################################################
#################### Plotting functions for Particle Filter ####################
################################################################################

#colorblind color palet (R,G,B)
black  = (0,    0,    0)
orange = (0.9,  0.7,  0)
lblue  = (0.35, 0.70, 0.90)
green  = (0,    0.6,  0.5)
yellow = (0.95, 0.90, 0.25)
dblue  = (0,    0.45, 0.70)
red    = (0.80, 0.40, 0)
pink   = (0.80, 0.60, 0.70)
    
def plot_fit(TITLE, ts, xh_data, xh_lv=None, xh_hv=None, UR_data=None, UR_lv=None, UR_hv=None, fig=None):
    #This plot_fit function plots the O2 [:,1], and CO2 [:,2] data

    #Setup the persistant figure 
    if fig is None:
        fig = plt.figure(figsize=(8,6))
        tax = fig.add_subplot(2, 1, 1)
        bax = fig.add_subplot(2, 1, 2, sharex=tax)
    else:
        tax = fig.subplot(2,1,1)
        bax = fig.subplot(2,1,2)

    tax.set_title(TITLE)
    tax.set_ylabel("Oxygen offgas[%]")
    tax2 = tax.twinx()
    tax2.set_ylabel("Carbon dioxide offgas[%]")
    bax.axhline(y=0, color='k', lw=0.5)
    bax.set_xlabel("Time (h)")
    bax.set_ylabel("Uptake rate [mol/s]")
    tax.set_xlim(ts[0], ts[-1])

    #Plot offgas data
    lnO2 = tax.plot(ts, xh_data[:, 1], 
            marker='o', markerfacecolor='none', markeredgewidth=1, 
            lw=0, color=dblue, label='xO2', alpha=0.5)
    lnCO2 = tax2.plot(ts, xh_data[:, 2], 
            marker='x', 
            lw=0, color=red, label='xCO2', alpha=0.5)

    if (xh_lv is not None) and (xh_hv is not None):
        tax.fill_between(ts, xh_lv[:,1], xh_hv[:,1], color=dblue, alpha=0.5)
        tax2.fill_between(ts, xh_lv[:,2], xh_hv[:,2], color=red, alpha=0.5)

    lns = lnO2+lnCO2
    labs = [l.get_label() for l in lns]
    tax.legend(lns, labs, loc=7)

    #Plot UR data if available
    if UR_data is not None:
        bax.plot(ts, UR_data[:,1], color=dblue, alpha=0.5, label='OUR')
        bax.plot(ts, UR_data[:,2], color=red, alpha=0.5, label='CUR')

    if (UR_lv is not None) and (UR_hv is not None):
        bax.fill_between(ts, UR_lv[:,1], UR_hv[:,1], color=dblue, alpha=0.5)
        bax.fill_between(ts, UR_lv[:,2], UR_hv[:,2], color=red, alpha=0.5)
        bax.legend(loc=0)

    #This plot returns an iterator that can be used to plot intermittent points during filtering
    t, estimate, var = yield fig.show()

    #Plot overlapping estimate data
    l, i = len(ts), 0
    lines = {10: tax.plot(ts, np.zeros(l)*np.nan, marker='o', ls='', color=dblue, alpha=0.5)[0], #O2_xh
             11: tax2.plot(ts, np.zeros(l)*np.nan, marker='o', ls='', color=red, alpha=0.5)[0], #CO2_xh
             1: bax.plot(ts, np.zeros(l)*np.nan, marker='.', ls='', color=dblue, alpha=0.5)[0], #O2_UR
             2: bax.plot(ts, np.zeros(l)*np.nan, marker='.', ls='', color=red, alpha=0.5)[0]}  #CO2_UR

    while True:
        for idx, line in lines.items():
            data_ = line.get_ydata()
            data_[i] = estimate[idx]
            line.set_ydata(data_)
            bax.relim()
            bax.autoscale_view(True,True,True)

        #fig.canvas.draw() #Some python backends require different redrawing steps
        #fig.canvas.flush_events()
        t, estimate, var = yield plt.pause(0.0000001)
        i+=1

def plot_final(ts, xh_data, xh_lv=None, xh_hv=None, UR_data=None, UR_lv=None, UR_hv=None, FL=None):
    #Setup the persistant figure 
    fig = plt.figure(figsize=(6,9))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

    #ax1.set_title(TITLE)
    ax1.set_ylabel("Oxygen offgas[%]")
    ax12 = ax1.twinx()
    ax12.set_ylabel("Carbon dioxide offgas[%]")

    ax2.axhline(y=0, color='k', lw=0.5)
    #m_ax.set_ylabel("Uptake rate [mol/s]") #TODO CHANGE TO [h]
    ax2.set_ylabel("Uptake rate [mmol/h]")

    #b_ax.set_ylabel("Total respired oxygen / carbon dioxode [mol]")
    ax3.set_ylabel("Cumulative [mmol]")
    #b_ax2 = b_ax.twinx()
    #b_ax2.set_ylabel("RQ [OU/CE]")
    ax4.set_xlabel("Time (h)")

    #t_ax.set_xlim(ts[0], ts[-1])
    ax1.set_xlim(-0.5, 12.5) #Are these good axes

    #Plot offgas data
    lnO2 = ax1.plot(ts, xh_data[:, 1], 
            marker='o', markerfacecolor='none', markeredgewidth=1, 
            lw=0, color=dblue, label='xO2', alpha=0.5)
    lnCO2 = ax12.plot(ts, xh_data[:, 2], 
            marker='x', 
            lw=0, color=red, label='xCO2', alpha=0.5)

    #Fill between TODO: add measurement uncertainty 
    ax1.fill_between(ts, xh_lv[:,1], xh_hv[:,1], color=dblue, alpha=0.5)
    lnO2fill = ax1.plot(np.NaN, np.NaN, color=dblue, lw=1, label='xO2 rec.', alpha=0.5)
    ax12.fill_between(ts, xh_lv[:,2], xh_hv[:,2], color=red, alpha=0.5)
    lnCO2fill = ax1.plot(np.NaN, np.NaN, color=red, lw=1, label='xCO2 rec.', alpha=0.5)

    lns = lnO2+lnCO2+lnO2fill+lnCO2fill
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)

    #Plot UR data if available
    #mol/s -> mmol/h
    f = 1000*3600
    ax2.plot(ts, UR_data[:,1]*f, color=dblue, alpha=1.0, label='OUR')
    ax2.plot(ts, UR_data[:,2]*f, color=red, alpha=1.0, label='CUR')

    ax2.fill_between(ts, UR_lv[:,1]*f, UR_hv[:,1]*f, color=dblue, alpha=0.5)
    ax2.fill_between(ts, UR_lv[:,2]*f, UR_hv[:,2]*f, color=red, alpha=0.5)
    ax2.legend(loc=0)

    #Plot Summation OUR/CER
    tsd = np.diff(ts) * 60 * 60
    tsd = np.append(tsd, tsd[-1])
    SUM = np.abs(np.cumsum(UR_data*tsd[:,None], axis=0))*1000 #TODO: Correct t=0 offset
    SUMh = np.abs(np.cumsum(UR_hv*tsd[:,None], axis=0))*1000 #TODO: Correct t=0 offset
    SUMl = np.abs(np.cumsum(UR_lv*tsd[:,None], axis=0))*1000 #TODO: Correct t=0 offset
    RQ = SUM[:,2]/SUM[:,1]

    lnSOUR = ax3.plot(ts, SUM[:,1], color=dblue, label=r'$\int_0^t ~~OUR~dt$', alpha=1.0)
    #ax3.fill_between(ts, SUMl[:,1], SUMh[:,1], color=dblue, alpha=0.3)
    lnSCUR = ax3.plot(ts, SUM[:,2], color=red, label=r'$\int_0^t ~-CUR~dt$', alpha=1.0)
    #ax3.fill_between(ts, SUMl[:,2], SUMh[:,2], color=red, alpha=0.3)
    lns = lnSOUR + lnSCUR
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc=7)

    #ax42 = ax4.twinx()
    green = "#5a9d35"
    ax4.set_ylabel("OUR [mmol/h]\nCumulative O2 [mmol]")
    ax4.plot(ts, UR_data[:,1]*f, color=green, label='OUR')
    ax4.plot(ts, SUM[:,1], color=dblue, label=r'$\int_0^t ~~OUR~dt$', alpha=1.0)
    ax4.set_ylim(*ax3.get_ylim())
    #ax4.legend(loc=0)
    ymax = ax4.get_ylim()[1]

    if FL is not None:
        #idx0 = ts.searchsorted(0)+2
        idxFL = ts.searchsorted(FL)
        idx0 = ts.searchsorted(FL/8)
        MAXO2 = max(SUM[:,1])
        FEASTO2 = SUM[idxFL,1]
        MAXOUR = max(UR_data[:,1])*f
        OUR0 = UR_data[idx0,1]*f

        ax4.axvline(x=FL, color="#b347c5", ls='dotted')
        #ax4.text(FL, 21-0.1, "feastlength", rotation=90, color='#b347c5', ha='right', va='top')
        ax4.text(FL, ymax-0.5, "feastlength", color='#b347c5', ha='center', va='top')

        ax4.axhline(y=MAXO2, color=dblue, ls='dotted', lw=1)
        ax4.text(12.4,MAXO2-0.5,'O2 total', color=dblue, va='top', ha='right')
        ax4.axhline(y=FEASTO2, color=dblue, ls='dotted', lw=1)
        ax4.text(12.4, FEASTO2-0.5,'O2 feast', color=dblue, va='top', ha='right')

        ax4.axhline(y=MAXOUR, color=green, ls='--', lw=0.5, xmax=0.5)
        ax4.text(6, MAXOUR,'OURmax', color=green, va='bottom', ha='center')
        ax4.axhline(y=OUR0, color=green, ls='--', lw=0.5, xmax=0.5)
        ax4.text(6, OUR0-0.5,'OUR0', color=green, va='top', ha='center')

    fig.subplots_adjust(top=0.98, bottom=0.06)
    fig.show()

    #plot_compound_profiles(ts, UR_data, FL)

    return fig

def plot_compound_profiles(ts, UR_data, FL, Ac0=None, fER=0.5, samples=None):
    if samples is None:
        samples = np.empty((4,4))*np.nan
    #Setup the persistant figure 
    fig = plt.figure(figsize=(6,7))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    UR_data[np.isnan(UR_data)] = 0

    #ax1.set_title(TITLE)
    ax1.set_ylabel("OUR [mmol/h]\nCumulative O2 [mmol]")

    #m_ax.set_ylabel("Uptake rate [mol/s]") #TODO CHANGE TO [h]
    ax2.set_ylabel("Compound amounts\n[mCmol]")

    ax3.set_ylabel("Biomass specific rates\n[mCmol/mCmolX/h]")
    ax3.set_xlabel("Time (h)")

    #t_ax.set_xlim(ts[0], ts[-1])
    ax1.set_xlim(-0.5, 12.5) #Are these good axes


    #Plot Summation OUR/CER
    tsd = np.diff(ts) * 60 * 60
    tsd = np.append(tsd, tsd[-1])
    SUM = np.abs(np.cumsum(UR_data*tsd[:,None], axis=0))*1000 #TODO: Correct t=0 offset

    f = 1000*3600 #mol/s -> mmol/h

    green = "#5a9d35"
    ax1.plot(ts, UR_data[:,1]*f, color=green, label='OUR')
    ax1.plot(ts, SUM[:,1], color=dblue, label=r'$\int_0^t ~~OUR~dt$', alpha=1.0)
    ax1.plot(ts, SUM[:,2], color=red, label=r'$\int_0^t ~~CER~dt$', alpha=1.0)
    #ax1.set_ylim(*ax3.get_ylim())
    ymax = ax1.get_ylim()[1]
    ax1.plot(ts, SUM[:,2]*np.nan, color='black', label='model', ls='--', alpha=1.0)
    ax1.legend(loc=0)

    #Find first datapoint after start of cycle
    idxFL = ts.searchsorted(FL)+1
    idx0 = ts.searchsorted(FL/8)
    idxEC = ts.searchsorted(11) #End of cycle at t ~ 11h
    MAXO2 = max(SUM[:,1])
    MAXO2 = max(SUM[:,2])
    FEASTO2 = SUM[idxFL,1]
    FEASTCO2 = SUM[idxFL,2]
    MAXOUR = max(UR_data[:,1])*f
    OUR0 = UR_data[idx0,1]*f
    Ac0 = Ac0 or np.sum(SUM[idxEC,:])


    ax1.axvline(x=FL, color="#b347c5", ls='dotted')
    Ac = Ac0*(1-SUM[:,1]/FEASTO2)
    Ac[:idx0-2] = 0
    Ac[Ac<0] = 0
    ax2.set_ylim((0,Ac0))

    FAMINEO2 = SUM[idxEC,1] - FEASTO2
    FAMINECO2 = SUM[idxEC,2] - FEASTCO2
    X0 = (Ac0 - SUM[idxEC,2])*(1/fER - 1)
    Xend = X0 + (Ac0 - SUM[idxEC,2])
    aO2 = (FEASTO2/Ac0-1/4)/(1/2.05-1/4)
    aCO2 = -(FEASTCO2/Ac0-1/3)/(1.05/2.05-1/3)
    a = (aO2 - aCO2)/2
    #XfeastO2 = (FEASTO2 - 1/4*Ac0)/(1-2.05*1/4) # Based on O2
    #XfeastCO2 = -(FEASTCO2 - 1/3*Ac0)/(1.05-2.05*1/3) # Based on CO2
    Xfeast = a * Ac0 * (1-1.05/2.05)
    Xfamine = Xend - Xfeast - X0

    X = np.ones(np.shape(ts)) * X0
    X[:idxFL] += Xfeast*(SUM[:idxFL,1]/FEASTO2) #Good till end of feast
    #X[:idxFL] += Xfeast*np.sum(SUM[:idxFL]/np.array([1,FEASTO2,FEASTCO2]), axis=1)/2
    X[idxFL:idxEC] += Xfeast + Xfamine * ((SUM[idxFL:idxEC,1]-FEASTO2)/FAMINEO2)
    #X[idxFL:idxEC] += Xfeast + Xfamine * np.sum((SUM[idxFL:idxEC]-[0,FEASTO2,FEASTCO2])/np.array([1,FAMINEO2,FAMINECO2]), axis=1)/2

    #((SUM[idxFL:idxEC,1]-FEASTO2)/FAMINEO2)

    X[idxEC:] = Xend
    X[idxEC:]*=(1-fER) #Dilute biomass

    PHB = np.zeros(np.shape(ts))
    #PHBfeastO2 = 2/3*(Ac0-FEASTO2*2.05)/(1-2.05*1/4)
    #PHBfeastCO2 = 2/3*(Ac0-FEASTCO2*2.05)/(1.05-2.05*1/3)
    PHBfeast = (1-a) * Ac0 * (1 - 1/3)
    PHB[:idxFL] = PHBfeast*(SUM[:idxFL,1]/FEASTO2) #Good till end of feast
    PHB[idxFL:idxEC] = PHBfeast*(1-(SUM[idxFL:idxEC,1]-FEASTO2)/FAMINEO2)
    PHB[idxEC:]*=(1-fER) #Dilute PHB

    ax2.plot(ts, Ac, ls='--', color='red', alpha=1.0, label='Ac')
    ax2.plot(ts, X, ls='--', color='purple', alpha=1.0, label='X')
    ax2.plot(ts, PHB, ls='--', color='orange', alpha=1.0, label='PHB')
    ax2.legend(loc=0)
    ax2.plot(samples[:,0]/60, samples[:,1], marker='^', ls=None, color='red')
    ax2.plot(samples[:,0]/60, samples[:,2], marker='.', ls=None, color='orange')
    ax2.plot(samples[:,0]/60, samples[:,3], marker='*', ls=None, color='purple')

    CO2 = np.zeros(np.shape(ts))
    CO2 = Ac0 - X + X0 - PHB - Ac
    CO2[:idx0-2] = 0
    CO2[idxEC:] = np.nan
    ax1.plot(ts, CO2, color='black', label='CO2', ls='--', lw=1.0, alpha=1.0)

    O2 = np.zeros(np.shape(ts))
    O2 = (Ac0 * 4 - (X - X0) * 4.2 - PHB * 4.5 - Ac * 4)/4
    O2[:idx0-2] = 0
    O2[idxEC:] = np.nan
    ax1.plot(ts, O2, color='black', label='O2', ls='--', lw=1.0, alpha=1.0)


    kd = OUR0/MAXOUR*(Xfeast+X0)/Xend*1/(1-fER)
    k = np.log(kd)/12
    print('Kd:\t',kd)
    X[:idxEC] = X[:idxEC] - max(0, 1-kd)*X0
    X[idxEC:] = X[idxEC:] - max(0, 1-kd)*X0*fER
    X *= np.exp(k*fER*ts)
    ax2.plot(ts, X, ls='--', color='#AAA', alpha=1.0, label='Xact')
    ax2.legend(loc=0)

    dAc = np.diff(Ac)
    dAc = np.append(dAc, dAc[-1])
    dAc[dAc>0] = 0
    dAcXdt = -dAc/tsd/X * 3600

    dX = np.diff(X)
    dX = np.append(dX, dX[-1])
    dX[dX<0] = 0
    dX[:idx0-2] = 0
    dXXdt = dX/tsd/X * 3600

    dPHB = np.diff(PHB)
    dPHB = np.append(dPHB, dPHB[-1])
    dPHB[:idx0-2] = 0
    dPHBXdt = dPHB/tsd/X * 3600

    ax3.plot(ts, dAcXdt, ls='--', color='red', label='$-q_s$')
    ax3.plot(ts, dXXdt, ls='--', color='purple', label='$\mu$')
    ax3.plot(ts, dPHBXdt, ls='--', color='orange', label='$q_{PHB}$')
    ax3.axhline(y=0, color='black', lw=0.5)
    ax3.legend(loc=0)

    # ax2.fill_between(ts, UR_lv[:,1]*f, UR_hv[:,1]*f, color=dblue, alpha=0.5)
    # ax2.fill_between(ts, UR_lv[:,2]*f, UR_hv[:,2]*f, color=red, alpha=0.5)

    # lnSOUR = ax3.plot(ts, SUM[:,1], color=dblue, label=r'$\int_0^t ~~OUR~dt$', alpha=1.0)
    # #ax3.fill_between(ts, SUMl[:,1], SUMh[:,1], color=dblue, alpha=0.3)
    # lnSCUR = ax3.plot(ts, SUM[:,2], color=red, label=r'$\int_0^t ~-CUR~dt$', alpha=1.0)
    # #ax3.fill_between(ts, SUMl[:,2], SUMh[:,2], color=red, alpha=0.3)
    # lns = lnSOUR + lnSCUR
    # labs = [l.get_label() for l in lns]
    # ax3.legend(lns, labs, loc=7)

    fig.subplots_adjust(top=0.98, bottom=0.06)
    fig.show()
    return fig


if __name__ == "__main__":
    print("Testing pickle loading of plotdata")
    import pickle, time

    R = 5
    fname = 'dataR{}'.format(R)
    #fname = 'testdata'
    Ac0 = [30.5, 33.5, 36, 36, 36, 36, 12.0, 35.0][R-1]

    try:
        samples = np.genfromtxt('measurements_{}.csv'.format(fname), delimiter=",")
    except:
        samples = None
    idx, dat, xh_lv, xh_hv, UR_mv, UR_lv, UR_hv, FL = pickle.load(open('{}.pickle'.format(fname), 'rb'))
    plot1 = plot_final(idx, dat, xh_lv, xh_hv, UR_mv, UR_lv, UR_hv, FL=FL)
    plot2 = plot_compound_profiles(idx, UR_mv, FL, Ac0, 0.5, samples)

    fpathname1 = 'images/plots/fig1-{}-{}'.format(fname, time.strftime('%Y%m%d-%H%M'))
    fpathname2 = 'images/plots/fig2-{}-{}'.format(fname, time.strftime('%Y%m%d-%H%M'))
    #plot1.savefig(fpathname1+".png", dpi=300)
    #plot2.savefig(fpathname2+".png", dpi=300)
