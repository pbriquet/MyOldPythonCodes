import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os, sys


def RC_to_V(rc):
    return np.power(6.24553e-3 - 1.08014e-4*rc + 4.32021e-7*np.power(rc,2),-1)
def BR_to_V(br):
    return 8.52592e-2 + 9.82889e-2*br + 1.89707e-4*np.power(br,2)
if __name__=='__main__':
    __location__=os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    ni_hardness = os.path.join(__location__,'Ni-based_hardness.xlsx')
    df = pd.read_excel(ni_hardness)
    df.replace(r'^\s*$', np.nan, regex=True)
    rows = df.dropna()
    rc = np.linspace(20.0,50.0,num=50)
    #rows = df.loc[(np.isnan(df['Brinell'])) & (np.isnan(df['Rockwell C']))]
    print(rows)
    p = np.polyfit(rows['Rockwell C'],rows['Brinell'],2)
    interpol = lambda x: p[0]*np.power(x,2) + p[1]*x + p[2]
    
    plt.plot(rc,interpol(rc),label='Polyfit',color='red')
    plt.plot(rows['Rockwell C'],rows['Brinell'],label='Table',color='b')
    print(p)
    plt.show()