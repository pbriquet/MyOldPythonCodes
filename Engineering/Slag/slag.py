from enum import IntEnum
from PeriodicTable import *

class Slag:
    def __init__(self,*tuple_oxide_masspercent):
        self.oxides = []
        self.mass_percent = []
        self.oxides_dict = dict()
        for i in tuple_oxide_masspercent:
            self.oxides.append(ChemicalCompound(i[0]))
            self.mass_percent.append(i[1])
            self.oxides_dict[i[0]]=i[1]
        self._calculate_molpercent()
        self._networkformers()
        self._basicity()
    def _calculate_molpercent(self):
        self.mol_percent = []
        tmp = 0.0
        for i in range(len(self.oxides)):
            tmp += self.mass_percent[i]/self.oxides[i].molecularmass
        for i in range(len(self.oxides)):
            self.mol_percent.append(self.mass_percent[i]/self.oxides[i].molecularmass/tmp)


    def _networkformers(self):
        f = 0.0
        # xT
        SiO2 = ChemicalCompound('SiO2')
        Al2O3 = ChemicalCompound('Al2O3')
        Fe2O3 = ChemicalCompound('Fe2O3')
        TiO2 = ChemicalCompound('TiO2')
        P2O5 = ChemicalCompound('P2O5')

        # y1NB
        CaO = ChemicalCompound('CaO')
        MgO = ChemicalCompound('MgO')
        FeO = ChemicalCompound('FeO')
        MnO = ChemicalCompound('MnO')
        Na2O = ChemicalCompound('Na2O')
        K2O = ChemicalCompound('K2O')

        

        tmp_xT = 0.0
        tmp_y1NB = 0.0
        tmp_y2NB = 0.0
        for i in range(len(self.oxides)):
            if(self.oxides[i] == SiO2):
                tmp_xT += self.mol_percent[i]
            elif(self.oxides[i] == Al2O3):
                tmp_xT += 2.0*self.mol_percent[i]
                tmp_y2NB -= 2.0*self.mol_percent[i]
            elif(self.oxides[i] == Fe2O3):
                tmp_xT += 2.0*f*self.mol_percent[i]
                tmp_y1NB -= 6.0*(1.0 - f)*self.mol_percent[i]
                tmp_y2NB -= 2.0*f*self.mol_percent[i]
            elif(self.oxides[i] == TiO2):
                tmp_xT += self.mol_percent[i]
            elif(self.oxides[i] == P2O5):
                tmp_xT += 2.0*self.mol_percent[i]
            elif(self.oxides[i] == CaO):
                tmp_y1NB += 2.0*self.mol_percent[i]
            elif(self.oxides[i] == MgO):
                tmp_y1NB += 2.0*self.mol_percent[i]
            elif(self.oxides[i] == FeO):
                tmp_y1NB += 2.0*self.mol_percent[i]
            elif(self.oxides[i] == MnO):
                tmp_y1NB += 2.0*self.mol_percent[i]
            elif(self.oxides[i] == Na2O):
                tmp_y1NB += 2.0*self.mol_percent[i]
            elif(self.oxides[i] == K2O):
                tmp_y1NB += 2.0*self.mol_percent[i]
        tmp_y2NB += tmp_y1NB

        self.xT = tmp_xT
        self.y1NB = tmp_y1NB
        self.y2NB = tmp_y2NB
        self.NBOT = self.y2NB/self.xT

    def _basicity(self):
        # Basic Oxides
        CaO = ChemicalCompound('CaO')
        MgO = ChemicalCompound('MgO')
        CaF2 = ChemicalCompound('CaF2')
        BaO = ChemicalCompound('BaO')
        SrO = ChemicalCompound('SrO')
        Na2O = ChemicalCompound('Na2O')
        K2O = ChemicalCompound('K2O')
        Li2O = ChemicalCompound('Li2O')
        MnO = ChemicalCompound('MnO')
        FeO = ChemicalCompound('FeO')
        P2O5 = ChemicalCompound('P2O5')

        # Acid Oxides
        SiO2 = ChemicalCompound('SiO2')
        Al2O3 = ChemicalCompound('Al2O3')
        TiO2 = ChemicalCompound('TiO2')
        ZrO2 = ChemicalCompound('ZrO2')

        basic_oxides = [(CaO,MgO,CaF2,BaO,SrO,Na2O,K2O,Li2O)]
        basic_oxides_half = [(MnO,FeO)]

        Mrajek_num = 0.0
        Mrajek_dem = 0.0
        Vee_num = 0.0
        Vee_dem = 0.0
        B_highMgOP2O5_num = 0.0
        B_highMgOP2O5_dem = 0.0
        B_unk_num = 0.0
        B_unk_dem = 0.0
        Blf_CaAl_num = 0.0
        Blf_CaAl_dem = 0.0
        Blf_num = 0.0
        Blf_dem = 0.0
        B_num = 0.0
        B_dem = 0.0

        for i in range(len(self.oxides)):
            if(self.oxides[i] == SiO2):
                Mrajek_dem += self.mass_percent[i]*self.oxides[i].atomweightfraction[1]
                Vee_dem += self.mass_percent[i]
                B_unk_dem += self.mass_percent[i]
                Blf_CaAl_dem += self.mass_percent[i]
                Blf_dem += self.mass_percent[i]
                B_dem += self.mass_percent[i]
                B_highMgOP2O5_dem += self.mass_percent[i]
            elif(self.oxides[i] == Al2O3):
                Blf_CaAl_dem += self.mass_percent[i]
                Blf_dem += 0.6*self.mass_percent[i]
                B_dem += 0.5*self.mass_percent[i]
            elif(self.oxides[i] == TiO2):
                Blf_CaAl_dem += self.mass_percent[i]
                B_dem += 0.5*self.mass_percent[i]
                B_unk_dem += self.mass_percent[i]
            elif(self.oxides[i] == ZrO2):
                B_dem += 0.5*self.mass_percent[i]
            elif(self.oxides[i] == P2O5):
                B_unk_dem += self.mass_percent[i]
                B_highMgOP2O5_dem + 0.84*self.mass_percent[i]

            elif(self.oxides[i] == CaO):
                Vee_num += self.mass_percent[i]
                Mrajek_num += self.mass_percent[i]
                B_highMgOP2O5_num += self.mass_percent[i]
                B_unk_num += self.mass_percent[i]
                Blf_CaAl_num += self.mass_percent[i]
                Blf_num += self.mass_percent[i]
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == MgO):
                B_highMgOP2O5_num += 1.4*self.mass_percent[i]
                B_unk_num += self.mass_percent[i]
                Blf_num += 1.4*self.mass_percent[i]
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == CaF2):
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == BaO):
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == SrO):
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == Na2O):
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == K2O):
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == Li2O):
                B_num += self.mass_percent[i]
            elif(self.oxides[i] == MnO):
                B_num += 0.5*self.mass_percent[i]
            elif(self.oxides[i] == FeO):
                B_num += 0.5*self.mass_percent[i]
        
        self.B = 1000.0 if B_dem==0.0 else B_num/B_dem
        self.B_highMgOP2O5 = 1000.0 if B_highMgOP2O5_dem==0.0 else B_highMgOP2O5_num/B_highMgOP2O5_dem
        self.Mrajek = 1000.0 if Mrajek_dem==0.0 else Mrajek_num/Mrajek_dem
        self.Vee = 1000.0 if Vee_dem==0.0 else Vee_num/Vee_dem
        self.Blf_CaAl = 1000.0 if Blf_CaAl_dem==0.0 else Blf_CaAl_num/Blf_CaAl_dem
        self.Blf = 1000.0 if Blf_dem==0.0 else Blf_num/Blf_dem
        self.B_unk = 1000.0 if B_unk_dem==0.0 else B_unk_num/B_unk_dem

    def Basicity(self):
        return [self.Mrajek,self.Vee,self.B_highMgOP2O5,self.B_unk,self.Blf_CaAl,self.Blf,self.B]
    def PrintBasicity(self):
        tmp = 'Mrajek = ' + str(self.Mrajek)
        tmp += "\nVee's Ratio = " + str(self.Vee)
        tmp += '\nB High MgO and P2O5 = ' + str(self.B_highMgOP2O5)
        tmp += '\nBlf (xCaO.yAl2O3) = ' + str(self.Blf_CaAl)
        tmp += '\nBlf = ' + str(self.Blf)
        tmp += '\nB = ' + str(self.B)
        return tmp


