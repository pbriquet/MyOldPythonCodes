import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import numpy as np

scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = scientific_match + '|' + float_match
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class JMatProQuenchingProperties:
    def __init__(self,filepath):
        quenchfile = open(filepath,'r')
        df = pd.read_csv(quenchfile,sep='\t')
        cols = df.columns
        cooling_rates = [float(re.search(r'TIME\(s\)\-Cooling_(' + float_match + r')',i).group(1)) for i in cols if re.search(r'TIME\(s\)\-Cooling_(' + float_match + r')',i) != None]
        phases = ['TOTAL']
        [phases.append(re.search(r'Phases vol\%\-(\S+)\-',i).group(1)) for i in cols if re.search(r'Phases vol\%\-(\S+)\-',i) != None and re.search(r'Phases vol\%\-(\S+)\-',i).group(1) not in phases]
        self.phases = phases
        replace = {'Thermal conductivity': 'K', 'Enthalpy': 'H', 'Poisson':'nu','Density':'rho','resistivity':'Omega','Electrical conductivity':'inv Omega','Hardness':'HRC','Specific heat':'Cp','Yield Stress':'LE','Tensile Stress':'LR','Molar volume':'Vm','Bulk modulus':'Kb','Shear modulus':'G','Young':'E','Phases vol':'x','Latent heat':'Lf'}
        self.data = dict()
        for i in cooling_rates:
            self.data[str(i)] = dict()
            for j in phases:
                cols_ij = ['T (C)'] + [k for k in cols if re.search(str(i),k) != None and re.search(j,k) != None]
                self.data[str(i)][j] = copy.copy(df[cols_ij])
                for k, v in replace.items():
                    for index, item in enumerate(cols_ij):
                        if(re.search(k,item) != None):
                            cols_ij[index] = v
                self.data[str(i)][j].columns = cols_ij

if __name__=='__main__':
    path = os.path.join(__location__,'cs130qp.dat')
    Jmat = JMatProQuenchingProperties(path)
    fig, ax = plt.subplots()
    print(Jmat.phases)
    property_to_plot = 'K'
    phase_to_plot = 'MARTENSITE'
    for cr,crv in Jmat.data.items():
        if(phase_to_plot in crv.keys()):
            if(property_to_plot in crv[phase_to_plot].columns):
                x = crv[phase_to_plot]['T (C)']
                y = crv[phase_to_plot][property_to_plot]
                ax.plot(x,y,label=cr)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, (end - start)/10))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, (end - start)/10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    plt.legend()
    plt.show()
'''
    for j in y.columns:
        
        
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, (end - start)/10))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, (end - start)/10))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        plt.title(j)
    plt.show()
    '''