class Compound:
    def __init__(self, name, charge, radius=4):
        self.name = name
        self.charge = charge
        self.radius = radius
        #self.set_activity_coefficient(ionic_strength=I, temp=T)
    
    def set_activity_coefficient(self, ionic_strength, T):
        """Determine the compound activity (gamma)"""
        #Depends on the temperature, charge, apparant radius, ionic strength, and pressure
        #TODO: implement pressure dependency (very slight)

        #Debye Huckel constants A & B at 25C  
        A,B = 0.5085, 0.3281        #REF: 10.2475/ajs.274.10.1199
        Tref = 25+273.15

        #Temperature correction
        T32, T12 = Tref**(3/2), Tref**(1/2)
        AT, BT = A * T32, B * T12
        A, B = AT / T**(3/2), BT / T**(1/2)
        
        sqrtI = ionic_strength**0.5
        self.gamma = 10**(-A*self.charge**2*sqrtI/(1+B*self.radius*sqrtI))
       
    def __repr__(self):
        return "{} activity {:.2}".format(self.name, self.gamma)
