import types
import numpy as np
from functools import reduce

from compound import Compound

#CONSTANTS
_allowedE = 1E-4        # Error margin for close to values
knowngasses = ('N2', 'H2', 'CO', 'O2', 'CO2', 'Ar', 'CH4')

DEFAULT_CONSTANTS = {'R'    : 8.314,
                     'kB'   : 1.38E-23,
                     'h'    : 6.63E-34,
                     'cH2O' : 1000/18.0,
                     'P'    : 1.0,          #Default pressure
                     'T'    : 298.15,       #Default temperature
                     'I'    : 0}            #Default ionic strength

ION_COMPOUNDS = {'H'      : Compound('H+',     1, 9.0),
                 'HCO3'   : Compound('HCO3-', -1, 4.0),
                 'CO3'    : Compound('CO3--', -2, 4.5),
                 'OH'     : Compound('OH-',   -1, 3.5)}

class ParticleFilterEnvironmentError(Exception):
    pass

class Environment:
    """Collection of environmental conditions of the bioreactor

    Instances can be called with a specific timepoint"""
    def __init__(self, gasnames):
        """Register base values for environmental conditions

        gasnames is a tuple of relevant gasses to the system"""
        self.gasnames = tuple(gasnames)
        self.n_gasses = len(self.gasnames)
        self.idx_by_name = dict(zip(self.gasnames, range(self.n_gasses)))

        self.co2speciation = 'CO2' in self.gasnames

        self.base_values = {}
        self.base_values.update(DEFAULT_CONSTANTS)
        self.base_values['ions'] = ION_COMPOUNDS

        self.current_values = {}
        self.triggers = {} 
        self.setters = {'ingas': self.set_gas,
                        'recgas': self.set_gas,
                        'T': self.set_temp,
                        'kLa': self.set_kLa}

    def __call__(self, t=None):
        """Get the environmental conditions at timepoint t"""

        updated = {}
        remove = []
        #Ideally sort triggers on t_start, t_end
        if t is not None:
            for name, (t_start, t_end, cb, once) in self.triggers.items():
                if t_start <= t < t_end:
                    updated.update(cb(self, name, t) or {})
                    if once: remove.append(name)

        for name in remove:
            self.unregister(name)

        ret = {}
        ret.update(self.base_values)
        ret.update(self.current_values)
        ret.update(updated)

        return ret

    def set(self, key, value, base=None):
        """Preferred method of setting environmental values"""
        if key in self.setters:
            value = self.setters[key](key, **value)
        if (base is True) or ((base is None) and (key not in self.base_values)):
            target = self.base_values
            #NOTE: You can set the base value without affecting the current value
        else:
            target = self.current_values
        target[key] = value

        if key in ['T', 'P', 'I']:
            print("Setting {}: ".format(key), end="")
            self.update() #Update dependent values

    def get(self, key, base=False):
        """Retrieve specific key from base or current values.

        This method is not preferred over env(t)[key] as it bypasses triggers"""
        if key in self.current_values and not base:
            return self.current_values[key]
        elif key in self.base_values:
            return self.base_values[key]
        else:
            raise ParticleFilterEnvironmentError("Keyword {} not present in environment".format(key))

    def register(self, name, t_start, t_end, call_back, once=False):
        """Allow a specific environmental condition to change between two timepoints"""
        self.triggers[name] = (t_start, t_end, call_back, once)

    def unregister(self, name):
        del(self.triggers[name])

    ######################################################################
    ######################## Verification Functions ######################
    ######################################################################

    def set_gas(self, name, Fn, x, dry=True):
        "Set the gasflow and molar composition of a specific gasstream"
        Pf = self.get('Pf')
        corr = (dry and 1.0) or Pf

        #Store dry and wet values for flow (F) and composition (x)
        Fdry = Fn * corr
        Fwet = Fdry / Pf

        if x is None: #No information on gas composition
            xdry = np.array([np.nan]*self.n_gasses)
        else:
            xdry = np.array(x) / corr
            lx = len(xdry)
            if lx != self.n_gasses:
                raise ParticleFilterEnvironmentError("Gas [{}] composition does not match registered gasses {} vs. {}.".format(name, lx, self.n_gasses))
            if not _close_to_one(sum(xdry)):
                raise ParticleFilterEnvironmentError("Gas [{}] dry molar-composition does not sum to 1.".format(name))

        xwet = xdry * Pf

        return {'Fdry': Fdry, 'Fwet': Fwet, 'xdry': xdry, 'xwet': xwet}

    def set_temp(self, name, celsius=None, fahrenheit=None, kelvin=None):
        if [celsius, fahrenheit, kelvin].count(None) != 2:
            raise ParticleFilterEnvironmentError("One temperature reference required:[Celsius ({}), Fahrenheit ({}) or Kelvin ({})]".format(celsius, fahrenheit, kelvin))
        if celsius is not None:
            kelvin = celsius + 273.15
        elif fahrenheit is not None:
            kelvin = (fahrenheit - 32)/1.8 + 273.15
        return kelvin

    def update(self):
        # Requires gasconstant R, planck constant h, boltzman constant kB
        # system pressure P, water molarity cH2O
        # Pw, Pf, Vm, co2-equilibria
        T = self.get('T')
        P = self.get('P')
        R = self.get('R')
        h = self.get('h')
        kB = self.get('kB')
        cH2O = self.get('cH2O')
        I = self.get("I")

        print("Updating dependent variables.")
        Tc = T - 273.15 
        Pw = 0.61121*np.exp((18.678-Tc/234.5)*(Tc/(257.15+Tc)))/100
        Pf = (1-Pw/P)
        Vm = R * T / P / 100    # L/mol
        pKw = 39.2598-0.13597*T+0.0001718*T**2 # REF: 10.1007/BF00811002
        Kw = 10**(-pKw)         # [M^2]
        self.set('Pw', Pw)
        self.set('Pf', Pf)
        self.set('Vm', Vm)
        self.set('Kw', Kw)

        #Set Henry's coefficients
        #print("Updating Henry's coefficients")
        henry_coeff = [get_Henry_coeff(gn, T) for gn in self.gasnames]
        self.set('H', np.array(henry_coeff))

        #CO2 speciation constants
        #print("Updating CO2 speciation constants")
        k1      = (kB*T/h)*np.exp((-79.2*1000-41.6*T)/(R*T))  # [1/s]
        k_1     = (kB*T/h)*np.exp((-69.16*1000+14.2*T)/(R*T)) # [1/s]
        K1      = k1/k_1

        k2      = (kB*T/h)*np.exp((-62.1*1000+40.4*T)/(R*T))  # [M/s]
        k_2     = (kB*T/h)*np.exp((-112.2*1000+63.7*T)/(R*T)) # [M/s]
        K2      = k2/k_2

        pKa2    = 2902.39/T-6.468+0.02379*T         # HCO3 - CO3 ~10.3
        Ka2     = 10**(-pKa2)

        #aCO2(aq) + aH2O <=> HCO3-(aq) + H+ K3 - REF: 10.1021/ja01250a059
        pK3     = 3404.71/T + 0.032786*T - 14.8435            # ~6.3
        K3      = 10**(-pK3)                                  # [M]
        Ka1     = K3/(cH2O*K1)
        pKa1    = -np.log10(Ka1)       

        co2constants = {
            'k1'   : k1,
            'k_1'  : k_1,
            'k2'   : k2,
            'k_2'  : k_2,
            'Ka1'  : Ka1,
            'Ka2'  : Ka2,
        }
        self.set('co2constants', co2constants)

        #print("Updating activity coefficients")
        for compound in self.base_values['ions'].values():
            compound.set_activity_coefficient(I, T)

    def set_kLa(self, name, kLaO2):
        kLa_values = [get_kLa(gn, kLaO2) for gn in self.gasnames]
        return np.array(kLa_values)

######################################################################
######################## END ENVIRONMENT CLASS #######################
######################################################################

######################################################################

######################################################################
######################## HENRY'S LAW CONSTANTS #######################
######################################################################
def get_Henry_coeff(gasname, T):
    """Retrieve Henry coefficient at a specific temperature

    Henry's law states that the amount of dissolved gas is proportional 
    to its partial pressure in the gas phase. (http://www.henrys-law.org/)"""
    #                          Hcp   dHcp (mol/m3Pa)
    H_parameters = {"CO2" : [3.3E-4, 2400],
                    "CO"  : [9.7E-6, 1300],  
                    "H2"  : [7.8E-6,  520],
                    "CH4" : [1.4E-5, 1900],
                    "N2"  : [6.4E-6, 1600],
                    "O2"  : [1.2E-5, 1700],
                    "Ar"  : [1.4E-5, 1700]}
    Tref = 298.15 #Kelvin

    Hcp, dHcp = H_parameters.get(gasname)
    return Hcp*np.exp(dHcp * (1/T - 1/Tref))*1E5/1000 #mol/L/bar

def get_kLa(gasname, kla_O2):
    """Retrieve relative kla to O2"""
    #NOTE:This requires much consideration and scrutiny 
    #Cussler, Edward Lansing, and Edward Lansing Cussler. Diffusion: mass transfer in fluid systems. Cambridge university press, 2009. (https://en.wikipedia.org/wiki/Mass_diffusivity#cite_note-Cussler-3)

    #Temperature dependence becomes relavant for some kLa's (Not for CO2?)
    #Patrick N. C. Royce and Nina F. Thornhill. Estimation of dissolved carbon dioxide concentrations in aerobic fermentations. AIChE Journal, 37(11):1680–1686, November 1991.

    #Small bubble systems have a 2/3 power for the diffusion ratio (Douwenga 2015)
    kLa_ratios = {"O2"  : [2.10, 2/3,   1],
                  "CO2" : [1.92, 2/3,   1],
                  "N2"  : [1.88, 2/3,   1],
                  "H2"  : [4.50, 2/3,   1],
                  "Ar"  : [2.00, 2/3,   1],
                  "CO"  : [2.03, 2/3,   1],
                  "CH4" : [1.49, 2/3,   1]}

    #There are other relationship formula, e.g.:
    #"H2"  : [0.28,   1,   1.29], 
    #http://refhub.elsevier.com/S1369-703X(15)00076-5/sbref0185

    DO2 = kLa_ratios["O2"][0]
    Dgas, p1, p2 = kLa_ratios[gasname]
    return (Dgas/DO2)**p1 * kla_O2**p2

######################################################################
########################## HELPER FUNCTIONS ##########################
######################################################################

def _close_to_zero(value):
    return (-_allowedE < value < _allowedE)

def _close_to_one(value):
    return _close_to_zero(value-1)

def _close_to_n(value, n):
    return _close_to_zero(value-n)

#TODO package / remove / implement #Currently not used
# Maybe something like QuantiPhy: https://quantiphy.readthedocs.io/en/stable/#
def gas_unit_conversion(unit_in, unit_out, Vm): #Vm in mol/l
    units = {'mol'  :1, 
             'mmol' :1/1000,
             'l'    :1/Vm,
             'ml'   :1/(1000*Vm),
             'h'    :1/3600,
             'min'  :1/60,
             'm'    :1/60,
             's'    :1}
    uin = unit_in.lower().split('/')
    uout = unit_out.lower().split('/')
    fin = reduce(lambda a,b: a*units[b], uin, 1)
    fout = reduce(lambda a,b: a*units[b], uout, 1)
    return fin/fout


######################################################################
######################## Enviroment setup test #######################
######################################################################

def test_environment():
    env = Environment(('N2', 'O2', 'CO2'))

    Vr      = 2.2           # [L]     Total Reactor Volume
    Vl      = 1.4           # [L]     Working volume
    Vb      = 0.05*Vl       # [L]     Bubble volume
    Vh      = Vr - Vl - Vb  # [L]     Headspace volume
    P       = 1             # [Bar]   Pressure

    env.set('Vr', Vr)
    env.set('Vl', Vl)
    env.set('Vb', Vb)
    env.set('Vh', Vh)
    env.set('P', P)

    #Set system temperature in any of the following ways
    Tc      = 30                    # [C]     Temperature in C
    env.set('T', {'celsius':Tc})
    #Tf      = 86                    # [F]     Temperature in F
    #env.set('T', {'fahrenheit':Tf})
    #T       = 273.15+Tc             # [K]     Temperature in K
    #env.set('T', {'kelvin':T})


    #TODO: temperature dependency
    env.set('kLa', {'kLaO2': 1.780/60})

    F_in = 300 #ml/min
    Fn_in = F_in/1000/60/env.get('Vm') #mol/s
    env.set('ingas', {'Fn':Fn_in, 'x':[0.7901, 0.2095, 0.0004]})

    F_rec = 0 #ml/min
    Fn_rec = F_rec/1000/60/env.get('Vm') #mol/s
    env.set('recgas', {'Fn':Fn_rec, 'x':None, 'dry':False})

    #Measurement stddev, smooth results
    env.set('meas_stddev',  np.array([400,    60,    20])/1E6)
    env.set('dUR',          np.array([0,      1.,    1.])/1E7)
    env.set('URcov',        np.diag( [1.,     1.,    1.]))

    return env

######################################################################
######################## Trigger testing funcs #######################
######################################################################

def Pf_test(env, name, t):
    "Only set different Pf during interval"
    return {"Pf": (0.99**t)*env.get('Pf')}

def Pf_test_permanent(env, name, t):
    "the callback function can set environmental values"
    env.set("Pf", (0.99**t)*env.get('Pf'))

def demonstrate_temperary_values():
    env = test_environment()
    #Standard value for Pf:
    print(env.get('Pf'))

    #Register Pf_test as alternative state between 10, 20 seconds
    env.register('New Pf', 10, 20, Pf_test)
    print(env(0)['Pf'])
    print(env(10)['Pf'])
    print(env(20)['Pf'])
    print(env(30)['Pf'])
    print(env(0)['Pf']) #Calls do not have to be time dependent

    #Overwrite value with trigger timeslot
    env.register('New Pf2', 40, np.inf, Pf_test_permanent, once=True)
    print(env(30)['Pf'])
    print(env(30)['Pf'])
    print(env(50)['Pf'])
    print(env(50)['Pf'])
    print(env(30)['Pf']) #Still prints out the new PF value
    print(env(30)['Pf']) #And does not roll back to previous env(30) values
    print(env.triggers)

def demonstrate_ionic_strength():
    env = test_environment()
    print(env.base_values['ions'])
    env.set('I', 0.05)
    print(env.base_values['ions'])
    env.set('T', {'celsius': 50})
    print(env.base_values['ions'])

if __name__ == "__main__":
    env = test_environment()
    print('kLa of [{}]: '.format(', '.join(env.gasnames)), env(0)['kLa'])

    #demonstrate_temperary_values()
    #demonstrate_ionic_strength()

