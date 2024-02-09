from AlloyDesign import *
import numpy as np
import matplotlib.pyplot as plt 

def CalculateMs_for_interdendritic():
    Ms_method = Martensite.Carapella.Ms
    alloy = Alloy(C=0.89,Mn=0.20,Si=0.15)
    Ms_0 = Ms_method(alloy,TScale=TempScale.Celsius)
    fs_max = 0.7
    fs_all_max = 0.99
    fs = np.linspace(0.0,fs_max,num=50)
    Cs = []
    Ms = []
    Cs_elements = {el:[] for el in alloy.elements.keys()}
    for _fs in fs:
        tmp = LiquidSteel.ScheilCs(alloy.elements,_fs)
        Cs.append(Alloy(**tmp))
        for k in alloy.elements.keys():
            Cs_elements[k].append(Cs[-1][k])
    fs_after = np.linspace(fs_max,fs_all_max,num=40)
    fs_all = list(fs)[:-1] + list(fs_after)
    
    Ms_ave = 0.0
    k_hold = 0
    for k,C in enumerate(Cs[1:]):
        Ms.append(Ms_method(C))
        Ms_ave += Ms[-1]*(fs_all[k+1] - fs_all[k])
        k_hold = k + 1
    Cl = Alloy(**LiquidSteel.ScheilCs(alloy.elements,fs_max))

    for k,_fs in enumerate(fs_after[1:]):
        Ms.append(Ms_method(Cl))
        Ms_ave += Ms[-1]*(fs_all[k_hold + k+1] - fs_all[k_hold + k])

    Ms_ave = Ms_ave/(fs_all[-1] - fs_all[0])

    '''
    for element in alloy.elements.keys():
        if(element != 'Fe'):
            plt.plot(fs,Cs_elements[element],label=element)
    '''
    #plt.legend()
    plt.plot(fs_all[1:],Ms,color='black')
    plt.hlines(Ms_0,fs_all[0],fs_all[-1],color='blue')
    plt.hlines(Ms_ave,fs_all[0],fs_all[-1],linestyle='--',color='red')
    plt.show()

def Calculate_Cementite():
    T = 293.15
    a,b,c = LatticeParameters.Cementite(293.15,TScale=TempScale.Kelvin)
    print(a,b,c)
    a = LatticeParameters.Alpha_Ferrite(293.15,TScale=TempScale.Kelvin)
    print(a)
    rho = (7875.96 - 0.297*T - 5.62e-5*T**2)*(1.0 - 2.62e-2*0.0)
    print(rho)
if __name__ == "__main__":
    Calculate_Cementite()

    