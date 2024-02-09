
import numpy as np
from enum import IntEnum
import pandas as pd
import os
from AlloySAPreader import *

class TCFE9:
    database_elements = ['VA','H','Mg','Ca','Y','Ti','Zr','Nb','Ta','Cr','Mo','W','Mn','Fe','Co','Ni','Cu','Zn','B','C','N','O','Al','Si','P','S','Ar','Ce']
    database_phases = ['GAS:G','LIQUID:L','BCC_A2','FCC_A1','HCP_A3','CBCC_A12','CUB_A13','DIAMOND_FCC_A4','BETA_RHOMBO_B','FC_ORTHORHOMBIC','RED_P','WHITE_P','GRAPHITE','CEMENTITE','M23C6','M7C3','M6C','M5C2','M3C2','MC_ETA','MC_SHP','KSI_CARBIDE','Z_PHASE','FE4N_LP1','FECN_CHI','PI','SIGMA','HIGH_SIGMA','MU_PHASE','P_PHASE','R_PHASE','CHI_A12','LAVES_PHASE_C14','M3SI','MN9SI2','MN11SI19','MN6SI','G_PHASE','FETI_B2','CR3SI','FE2SI','MSI','M5SI3','MG2NI','MG2SI','NBNI3','NITI2','NI3TI','NIY','NI17Y2','NI2Y','NI2Y3','NI3Y','NI4Y','NI7ZR2','NI5Y','ALY','AL2Y3','ALY2','AL3Y_HT','AL3Y_LT','AL2Y_C15','CO17Y2','CO5Y_D2D','CO3Y','CO3Y2','CO7Y6','COY_BF','CO3Y4','CO5Y8','YSI2_LT','YSI2_HT','Y3SI5_HT','Y3SI5_LT','Y5SI4','MSI_B27','CUZR_B2','CU2Y_H','CU2Y_L','CU6Y','CU4Y','CU7Y2','D022_AL3NB','D019_CO3MO','CO3VV','MNTA','MOSI2_C11B','MO5SI3_D8M','GAMMA2_ALFEZN','FE2SITI_L21','AL8MN5_D810','KAPPA_E21','AL4C3','B4C','M12C','FE8SI2C','MGC2','SIC','MN5SIC','YC2_C11A','Y15C19_H','Y15C19_R','Y2C3_H','Y2C3_R','YC_GAMMA','MZR3_E1A','K_PHASE','CA3ZN','CA5ZN3','ZNCA','ZN2CA','CAZN3','CAZN5','CAZN11','CAZN13','COZN4_GAMMA','COZN_GAMMA1','COZN_GAMMA2','COZN_BETA1','CRZN17','GAMMA_D82','GAMMA1_FEZN','DELTA_FEZN','ZETA_FEZN','CUZN_EPSILON','CU5ZN8_GAMMA_D83','MG2ZN3','MGZN','MG51ZN20','MG2ZN11','MNZN9','MOZN7','MOZN22','NIZN8_DELTA','BETA1','GAMMA','ZN3P2','ZNP2','ZNS_ALPHA_B3','ZNS_BETA_B4','TI2ZN','TIZN1','TIZN2','TIZN3','TIZN5','TIZN10','TIZN15','A_YZN2','B_YZN2','YZN3','Y3ZN11','Y13ZN58','YZN5','Y2ZN17','YZN12','YZN','ZN22ZR1','ZN39ZR5','ZN3ZR1L','ZN3ZR1H','ZN2ZR1','ZN1ZR1','ZN2ZR3','ZN1ZR2','ALMGZN_T2','ALMGZN_Q','ALMG_BETA','ALMG_EPSILON','AL12MG17_A12','ALMGZN_PHI','ALMGZN_T1','AL2FE1','AL5FE4','AL5FE2','AL13FE4','AL7CR','AL2CR3','TAU4_ALCRZN','ALN','BN_HP4','SI3N4','TI2N','MN6N4','MN6N5','TAN_EPS','MB_B27','MB_B33','MOB_BG','M2B_C16','M2B_CB','M3B2_D5A','NB3B2_D5A','MB2_C32','MOB2','B3SI_D1G','M2B3','CR5B3_D8L','YB4_D1E','M3B4_D7B','MO2B5_D8I','YB6_D21','M5B6','YB12_D2F','YB66','FE1NB1B1_C22','FE3NB3B4','FEWB_C37','FE5SIB2','FE10SI2B3','FE5SI2B','MP_B31','CU3P_D021','M2P_C22','M3P_D0E','MN3P_D0E','MO3P_D0E','MOP_BH','TI3P','FENBP','FENB2P','FENB4P','NBP','NB7P4','FETIP','FESI4P4','ALP','SIP','SIP2','CORUNDUM:I','HALITE:I','SPINEL:I','ALPHA_SPINEL:I','MN1O2:I','NIMNO3:I','NI6MNO8_TYPE:I','QUARTZ','CRISTOBALITE','TRIDYMITE','RHODONITE:I','OLIVINE:I','CAMNO3:I','CAMN2O4:I','CA1CR2O4_A','CA1CR2O4_B','C1A1:I','AF','CF:I','C1A2:I','CF2:I','C1A6:I','C2F:I','C3A1:I','CWF:I','CW3F:I','C4WF4:I','C4WF8:I','C3A2M1:I','C1A8M2:I','C2A14M2','CORDIERITE','SAPPHIRINE','CA2SIO4_ALPHA:I','CA2SIO4_ALPHA_PRIME:I','LARNITE:I','LOWCLINO_PYROXENE:I','CLINO_PYROXENE:I','ORTHO_PYROXENE:I','PROTO_PYROXENE:I','WOLLASTONITE:I','PSEUDO_WOLLASTONITE:I','ANDALUSITE:I','SILLIMANITE:I','MULLITE:I','KYANITE:I','HATRURITE:I','MELILITE:I','RANKINITE:I','ANORTHITE:I','MERWINITE:I','TIO:I','TIO_ALPHA:I','TI3O2:I','RUTILE_MO2:I','NBO:I','FE4NB2O9:I','AL2TIO5:I','FLUORITE_C1:I','ZRO2_TETR:I','ZRO2_MONO:I','S2ZR1','M2O3C:I','M2O3H:I','YAG:I','YAP:I','YAM:I','Y2S2D_Y2SI2O7','Y2S2G_Y2SI2O7','Y2S2B_Y2SI2O7','Y2S2A_Y2SI2O7','Y2SIO5','ZR3Y4O12','MN2YO5','MNYO3_HEX','CUPRITE_C3:I','CUO','YCUO2','Y2CU2O5','YFE2O4','PYRRHOTITE','PYRITE','DIGENITE','MNS','NI3S2','TI4C2S2','FESO4','FE2S3O12','SIS2','AL2S3','ZR2S3','COVELLITE','ALPHA_CHALCOCITE','BETA_CHALCOCITE','DJURLEITE','ANLITE','CEZN','CEZN2','CE2ZN17','CEZN11','CE7NI3','CEZN3','CE3ZN11','CE13ZN58','CEZN5','CE3ZN22','CENI','CENI2','CENI3','CE2NI7','CENI5','MG2CE','MG41CE5','MG3CE','MG17CE2','MG12CE','CO5CE','CO19CE5','CO7CE2','CO3CE','CO2CE','CO11CE24','AL4CE','AL1CE3','AL3CE_BETA','AL3CE_ALPHA','AL2CE','ALCE','ALCE2','ALCE3_BETA','ALCE3_ALPHA','CE2FE17','CEFE2','A_CE2O3:I','H_CE2O3:I','X_CE2O3:I','C_CE2O3:I','F_CEO2:I','CE7O12','CE1S1','CE1S2','CE2S3','CE3S4','CESI2','CE2O12S3','CE2O2S1','B4CE','B6CE','CEC2_BETA','CEC2_ALPHA','CU6CE','CU5CE','CU4CE','CU2CE','CUCE','CE2C3','CE1N1']
    balance = 'Fe'

class TCExporter:
    # Elementos possíveis no TCFE9
    TCFE9_elements = ['VA','H','Mg','Ca','Y','Ti','Zr','Nb','Ta','Cr','Mo','W','Mn','Fe','Co','Ni','Cu','Zn','B','C','N','O','Al','Si','P','S','Ar','Ce']
    def __init__(self,comp,folder_name,phases=None,residuals=None,with_residuals=False):
        self.comp = comp
        self.folder_name = folder_name
        try:
            os.stat(folder_name)
        except:
            os.mkdir(folder_name)
        self.elements = list(comp.keys())
        self.phases = phases
        self.residuals = residuals
        self.with_residuals = with_residuals

    def eqNominal(self):
        folder = os.path.realpath(os.path.join(self.folder_name,'Eq'))
        try:
            os.stat(folder)
        except:
            os.mkdir(folder)
        filepath = os.path.join(folder,'pm.tcm')
        TCExporter.createMacro(filepath,self.elements,self.comp,self.phases)
    def eqVariation(self,elements_variation,foldername='EqVar'):
        folderpath = os.path.join(self.folder_name,foldername)

        try:
            os.stat(folderpath)
        except:
            os.mkdir(folderpath)
        #log_file = open(os.path.join(folderpath,'log.dat'),'w')
        log_pd = pd.DataFrame(columns=['name'] + self.elements)
        comps = []
        for k,v in elements_variation.items():
            comps.append(np.linspace(v[0],v[1],num=v[2]))
        mesh = np.meshgrid(*comps,indexing='ij')
        mesh_dict = {}
        j = 0
        for k,v in elements_variation.items():
            mesh_dict[k] = mesh[j].ravel()
            j+=1
        size = mesh[0].size 
        
        for c in range(1,size+1):
            folder = os.path.join(folderpath,str(c))
            try:
                os.stat(folder)
            except:
                os.mkdir(folder)
            comp = {}
            for k,v in elements_variation.items():
                comp[k] = mesh_dict[k][c-1]
            for k,v in self.comp.items():
                if(k not in comp):
                    comp[k] = v
            #TCExporter.createMacro(os.path.join(folder,'eq.tcm'),self.elements,comp,self.phases)
            comp['name'] = str(c)
            log_pd = log_pd.append(comp,ignore_index=True)
        log_pd.to_csv(os.path.join(folderpath,'log.csv'),index=False,sep=';')


    def scheilNominal(self,fast_diffusion=None):
        folder = os.path.realpath(os.path.join(self.folder_name,'Scheil'))
        try:
            os.stat(folder)
        except:
            os.mkdir(folder)
        filepath = os.path.join(folder,'scheil.tcm')
        TCExporter.createScheilMacro(filepath,self.comp,self.phases,fast_diffusion)
    def scheilVariation(self,fast_diffusion=None):
        folder = os.path.realpath(os.path.join(self.folder_name,'Scheil Variations'))
        try:
            os.stat(folder)
        except:
            os.mkdir(folder)
        filepath = os.path.join(folder,'scheil.tcm')
        TCExporter.createScheilMacro(filepath,self.comp,self.phases,fast_diffusion)


    @staticmethod
    def createMacro(filepath,elements,comp,phases):
        mfile = open(filepath,'w')
        mfile.write('go data\n')    # (Goto-module) Entra em modo Data
        mfile.write('sw tcfe9\n')   # (Switch-database) Database TCFE9
        mfile.write('d-sys Fe')     # (Define-System) Definir elementos, inicial mantem o balanço
        for i in elements:
            mfile.write(' ' + i)    # Adiciona elementos a serem considerados
        mfile.write('\nrej p *\n')  # (Reject phase) Rejeita todas as fases

        mfile.write('rest p\n')     # (Rest phase) Adiciona fases
        for i in phases:
            mfile.write(' ' + i)    # Adiciona fases da lista
        mfile.write('\n\n\n')
        mfile.write('get\n')        # (Get) Inicia sistema
        mfile.write('go p-3\n')     # (Goto-module) Poly-3
        mfile.write('s-c p=1e5 n=1 t=2000\n\n') # Set-Define-Axis X para temperatura em Celsius
        mfile.write('ADVANCED_OPTIONS GLOBAL_MINIMIZATION Y,,,,,,,,\n\n')   # Liga modo de global minization
        for k,v in comp.items():
            mfile.write('s-c w(' + k + ')=' + str(v) + 'E-02\n')    # Adiciona os elementos na composição
        
        mfile.write('\n\n')
        for i in range(5):  # 5x Calculate equilibrium
            mfile.write('c-e\n')    # Calculate equilibrium
        mfile.write('\nl-eq,,,,,\n\n') # Set-Define-Axis X para temperatura em Celsius
        mfile.write('s-a-v 1 t 400 2000 5\n') # (Set-Axis-Variable) 1 = X, t = temperatura, range entre 400 e 2000, de 5 em 5
        mfile.write('add 1 2 -1 -2\n\n') # (Add)
        mfile.write('step,,,,\n\n') # Step
        mfile.write('pos\n\n')  # Post-processing
        mfile.write('s-d-a x t-c\n')    # Set-Define-Axis X para temperatura em Celsius
        mfile.write('s-d-a y bpw(*) *\n\n') # Set-Define-Axis Y porcentagem em peso de fases
        mfile.write('s-a-ty y log\n') # (Set-Axis-Type) Y em escala logaritmica
        mfile.write('s-s-s x n 400 1600\n') # (Set-Scale) X variar entre 400 e 1600
        mfile.write('s-s-s y n 1e-4 1 \n\n') # (Set-Scale) Y variar entre 1e-4 e 1
        mfile.write('s-l f\n\n') # (Set-Layout) Sistemas de cores para visualização no ThermoCalc
        mfile.write('make-exp-data file bpw.exp\n\n') # Faz arquivo de saída que será lido pelo Python em seguida

    @staticmethod
    def createScheilMacro(filepath,comp,phases,fast_diffusion=None):
        mfile = open(filepath,'w')
        mfile.write('go scheil\n')
        mfile.write('start_wizard\n')
        mfile.write('TCFE9\n')
        mfile.write('Fe\n')
        mfile.write('Y\n')
        for k,v in comp.items():
            mfile.write(k +'\n')
            mfile.write(str(v) + '\n')
        mfile.write('\n2000\n')
        
        if(phases==None):
            mfile.write('NONE\n')
        else:
            mfile.write('*\n')
            for i in phases:
                mfile.write(i + '\n')
        mfile.write('NONE\n')   # End Phases
        mfile.write('Y\n')
        mfile.write('N\n')
        if(fast_diffusion==None):
            mfile.write('NONE')   # Fast Diffusion
        else:
            for j in fast_diffusion:
                mfile.write(j + ' ')
        mfile.write('\nY\n')
        mfile.write('N\n')
        mfile.write('NONE\n')
        #mfile.write('bs\n')
        #mfile.write('t\n')
        mfile.write('make-exp-data file scheil.exp\n')
        mfile.write('set-inter\n\n')
        mfile.write('set-inter\n\n')
        mfile.write('exit\n')
        mfile.close()

def test_VPMM():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    folder_path = os.path.join(__location__,'Test')
    phases = ['LIQUID','FCC_A1','BCC_A2','CEM','M7C3','M23C6','HCP_A3','M6C']
    comp = {'C':0.2,'Si':0.35,'Mn':0.77,'Cr':1.48,'Mo':3.3,'Ni':2.41,'V':0.01,'Co':0.03,'W':1.56,'Cu':0.074,'N':0.0072,'Al':0.0063}
    elements_variation = {'C':(0.15,0.25,5),'Si':(0.25,0.45,5),'Mn':(0.7,0.85,5)}
    tc = TCExporter(comp,folder_path,phases=phases)
    tc.eqVariation(elements_variation)

def test_FeCSi():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    folder_path = os.path.join(__location__,'FeCSi')
    phases = ['LIQUID','FCC_A1','BCC_A2','CEM']
    comp = {'C':0.15,'Si':0.35}
    elements_variation = {'C':(0.15,0.8,50)}
    tc = TCExporter(comp,folder_path,phases=phases)
    tc.eqVariation(elements_variation)
if __name__=='__main__':

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    folder_path = os.path.join(__location__,'Bonon')
    print(folder_path)
    input()
    phases = ['LIQUID','FCC_A1','BCC_A2','CEM','M7C3','M23C6','HCP_A3','M6C']
    comp = {'C':0.5,'Si':0.7,'Mn':0.4,'Cr':1.48,'Mo':1.0,'S':0.03,'Ni':0.4,'Cr':3.0,'V':1.7,'Nb':0.1,'W':3.4,'Cu':0.3,'N':0.001,'Al':0.003}
    tc = TCExporter(comp,folder_path,phases=phases)
    tc.scheilNominal()
