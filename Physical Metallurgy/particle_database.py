class HardParticle:
    
    def __init__(self,density,hardness):
        self.hardness = hardness
        self.density = density

class HardPaticlesDatabase:
    def __init__(self):
        self.phases = {}
        self.phases['TiC'] = HardParticle(4.9,0.0)
        self.phases['SiC'] = HardParticle(4.9,0.0)
        self.phases['TiC'] = HardParticle(4.9,0.0)
        self.phases['TiC'] = HardParticle(4.9,0.0)